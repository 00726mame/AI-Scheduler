import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
import pickle
import os
import math
from tqdm import tqdm

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPISODES = 200
STEPS_PER_EPISODE = 60
CONTEXT_LENGTH = 128
SAVE_PATH = "trajectories.pkl"

class WikipediaDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
        
    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []
        while True:
            try:
                item = next(iterator)
                text = "タイトル:\n" + item["title"] + "\n\n本文:\n" + item["text"]
                tokenized = self.tokenizer(text, add_special_tokens=True)['input_ids']
                buffer.extend(tokenized)
                
                while len(buffer) >= self.max_length:
                    chunk = buffer[:self.max_length]
                    buffer = buffer[self.max_length:]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": input_ids, "labels": input_ids.clone()}
            except StopIteration:
                break

class TrainingEnv:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    def step(self, params, batch):
        """
        Applies params, runs one training step, returns (loss, grad_norm, weight_norm).
        """
        # Apply Params
        for param_group in self.optimizer.param_groups:
            if "lr" in params:
                param_group["lr"] = params["lr"]
            if "weight_decay" in params:
                param_group["weight_decay"] = params["weight_decay"]
        
        # Max Grad Norm is applied during step usually, but here manually
        max_grad_norm = params.get("max_grad_norm", 1.0)
        
        # Forward
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Grad Norm
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        self.optimizer.step()
        
        # Weight Norm
        weight_norm = 0.0
        for p in self.model.parameters():
            weight_norm += p.data.norm(2).item() ** 2
        weight_norm = weight_norm ** 0.5
        
        return loss.item(), total_norm.item(), weight_norm

# Utils for State Management
def recursive_clone(obj):
    if isinstance(obj, torch.Tensor): 
        return obj.clone()
    elif isinstance(obj, dict): 
        return {k: recursive_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list): 
        return [recursive_clone(v) for v in obj]
    else: 
        return obj

def recursive_update(src, dst):
    if isinstance(src, dict) and isinstance(dst, dict):
        for k, v in src.items():
            if k in dst:
                if isinstance(v, torch.Tensor) and isinstance(dst[k], torch.Tensor):
                    dst[k].copy_(v)
                elif isinstance(v, (dict, list)):
                    recursive_update(v, dst[k])
                else:
                    dst[k] = v
            else:
                dst[k] = recursive_clone(v)
    elif isinstance(src, list) and isinstance(dst, list):
        for i, v in enumerate(src):
            if i < len(dst):
                if isinstance(v, torch.Tensor) and isinstance(dst[i], torch.Tensor):
                    dst[i].copy_(v)
                elif isinstance(v, (dict, list)):
                    recursive_update(v, dst[i])
                else:
                    dst[i] = v
            else:
                dst.append(recursive_clone(v))

def get_action_values(action):
    """Map normalized action to hyperparameters."""
    action = np.clip(action, -1.0, 1.0)
    # LR: -1.0 -> 1e-6, 0.0 -> 3e-5, 1.0 -> 1e-3
    lr_log = action[0] * 1.5 - 4.5
    lr = 10 ** lr_log
    
    weight_decay = (action[1] + 1) / 2 * 0.1
    max_grad_norm = (action[2] + 1) / 2 * 4.9 + 0.1
    return {"lr": lr, "weight_decay": weight_decay, "max_grad_norm": max_grad_norm}

def collect_data(episodes=EPISODES, save_path=SAVE_PATH):
    print(f"🚀 Starting Data Collection ({episodes} episodes)...")
    
    # Tiny Model
    config = GPT2Config(
        n_positions=CONTEXT_LENGTH,
        n_ctx=CONTEXT_LENGTH,
        n_embd=128,
        n_layer=2,
        n_head=4,
        vocab_size=50257
    )
    # Use standard tokenizer (requires internet to download vocab)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Suppress warning about token length (we chunk manually)
    tokenizer.model_max_length = 100000
    
    dataset = WikipediaDataset(tokenizer, max_length=CONTEXT_LENGTH)
    dataloader = DataLoader(dataset, batch_size=8)
    data_iterator = iter(dataloader)
    
    all_trajectories = []
    
    for ep in tqdm(range(episodes)):
        # Initialize new model for each episode to simulate fresh start
        model = GPT2LMHeadModel(config).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Initial dummy
        env = TrainingEnv(model, optimizer, DEVICE)
        
        # Buffer for state snapshot
        buffer_model_state = recursive_clone(model.state_dict())
        buffer_opt_state = recursive_clone(optimizer.state_dict())

        # Get initial batch
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
            
        # Initial Loss
        with torch.no_grad():
            outputs = model(batch["input_ids"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            current_loss = outputs.loss.item()
            
        ema_loss = current_loss
        prev_loss = current_loss
        
        # Episode Buffers
        obs_buf, act_buf, rew_buf = [], [], []
        
        # Initial Action (Neutral)
        current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # variables to track for state
        grad_norm = 0.0
        weight_norm = 0.0
        
        for step in range(STEPS_PER_EPISODE):
            # Calculate derived features
            delta_loss = current_loss - prev_loss
            
            # Re-calculate params to get LR
            current_params_state = get_action_values(current_action)
            current_log_lr = np.log10(current_params_state["lr"] + 1e-9)

            # State: [loss, ema_loss, grad_norm, weight_norm, progress, log_lr, delta_loss]
            progress = step / STEPS_PER_EPISODE
            state = np.array([
                current_loss, 
                ema_loss, 
                grad_norm, 
                weight_norm, 
                progress,
                current_log_lr,
                delta_loss
            ], dtype=np.float32)
            
            # --- Best-of-N Candidate Selection ---
            
            # Snapshot current state
            recursive_update(model.state_dict(), buffer_model_state)
            recursive_update(optimizer.state_dict(), buffer_opt_state)

            # Candidates: [Increase LR, Keep, Decrease LR]
            candidates_cfg = [
                np.clip(current_action + np.array([0.5, 0.0, 0.0]), -1.0, 1.0).astype(np.float32),
                np.clip(current_action + np.array([0.1, 0.0, 0.0]), -1.0, 1.0).astype(np.float32),
                current_action,
                np.clip(current_action - np.array([0.5, 0.0, 0.0]), -1.0, 1.0).astype(np.float32),
                np.clip(current_action - np.array([0.1, 0.0, 0.0]), -1.0, 1.0).astype(np.float32)
            ]
            
            # Get next batch for evaluation
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch = next(data_iterator)

            best_result = None
            best_reward = -float('inf')

            # Epsilon-Greedy Exploration
            # If random < 0.2, just pick one randomly without simulation? 
            # OR simulate all but pick random? 
            # User wants "Select Best", so we should Simulate All.
            # But maybe add noise to candidates?
            
            # Let's clean candidates: Add noise?
            # 1_collect_data does: momentum, local(+noise), global.
            # Here we keep simple set.
            
            evaluated_candidates = []

            for cand_action in candidates_cfg:
                # Add small noise for diversity
                cand_action = np.clip(cand_action + np.random.normal(0, 0.05, size=3), -1.0, 1.0).astype(np.float32)
                
                # Restore state
                model.load_state_dict(buffer_model_state)
                optimizer.load_state_dict(buffer_opt_state)
                
                params = get_action_values(cand_action)
                
                try:
                    c_loss, c_grad_norm, c_weight_norm = env.step(params, batch)
                    
                    if math.isnan(c_loss) or math.isinf(c_loss):
                        c_loss = 20.0
                        
                    # Calculate Reward
                    # 1. Improvement
                    rel_improvement = (prev_loss - c_loss) / (prev_loss + 1e-6)
                    r_im = rel_improvement * 10.0 # Scale
                    
                    # 2. Stability (Grad Norm)
                    if math.isnan(c_grad_norm): c_grad_norm = 100.0
                    r_st = 0.5 if c_grad_norm < 2.0 else -0.5 # Simple threshold reward
                    # User asked to consider grad_norm
                    
                    # 3. Absolute Loss
                    r_abs = max(0, 2.0 - c_loss) * 0.5
                    
                    total_reward = r_im + r_st + r_abs
                    
                    # Store
                    res = {
                        "action": cand_action,
                        "loss": c_loss,
                        "grad_norm": c_grad_norm,
                        "weight_norm": c_weight_norm,
                        "reward": total_reward,
                        "model_state": recursive_clone(model.state_dict()),
                        "opt_state": recursive_clone(optimizer.state_dict())
                    }
                    evaluated_candidates.append(res)
                    
                except Exception as e:
                    pass

            # Selection
            if len(evaluated_candidates) > 0:
                # Epsilon-Greedy on SELECTION
                if random.random() < 0.1:
                    # Random selection from evaluated to enable exploration
                    selected = random.choice(evaluated_candidates)
                else:
                    # Greedy selection
                    evaluated_candidates.sort(key=lambda x: x["reward"], reverse=True)
                    selected = evaluated_candidates[0]
                
                # Apply selected state
                current_action = selected["action"]
                loss = selected["loss"]
                grad_norm = selected["grad_norm"]
                weight_norm = selected["weight_norm"]
                reward = selected["reward"]
                
                # Restore the selected state to be the current state
                model.load_state_dict(selected["model_state"])
                optimizer.load_state_dict(selected["opt_state"])
                
            else:
                # Fallback if all failed
                loss = prev_loss
                reward = -2.0
                model.load_state_dict(buffer_model_state)
                optimizer.load_state_dict(buffer_opt_state)

            # Update State Trackers
            ema_loss = 0.9 * ema_loss + 0.1 * loss
            prev_loss = current_loss 
            current_loss = loss
            
            obs_buf.append(state)
            act_buf.append(current_action)
            rew_buf.append(reward)
            
        all_trajectories.append({
            "observations": np.array(obs_buf),
            "actions": np.array(act_buf),
            "rewards": np.array(rew_buf),
            "dones": np.zeros(len(obs_buf))
        })
        
    with open(save_path, "wb") as f:
        pickle.dump(all_trajectories, f)
        
    print(f"Saved {len(all_trajectories)} episodes to {save_path}")

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--save_path", type=str, default="trajectories.pkl")
    args = parser.parse_args()
    
    collect_data(episodes=args.episodes, save_path=args.save_path)
    # Force exit to avoid PyGILState_Release fatal errors due to threads
    sys.exit(0)
