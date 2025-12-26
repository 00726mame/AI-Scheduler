import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DecisionTransformerConfig, DecisionTransformerModel, get_linear_schedule_with_warmup

try:
    from .utils import get_statistics, normalize_trajectories
except ImportError:
    from utils import get_statistics, normalize_trajectories

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, context_length=20):
        self.trajectories = trajectories
        self.context_length = context_length
        self.indices = []
        for i, traj in enumerate(self.trajectories):
            T = len(traj['rewards'])
            for t in range(T):
                self.indices.append((i, t))
                
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, end_time = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        start_time = max(0, end_time - self.context_length + 1)
        real_len = end_time - start_time + 1
        
        states = traj['observations'][start_time : end_time + 1]
        actions = traj['actions'][start_time : end_time + 1]
        rtg = traj['rtg'][start_time : end_time + 1]
        timesteps = np.arange(start_time, end_time + 1)
        
        def pad(arr, pad_val=0):
            if len(arr) < self.context_length:
                pad_len = self.context_length - len(arr)
                if arr.ndim == 1:
                    return np.concatenate([np.full(pad_len, pad_val), arr])
                else:
                    bs = np.full((pad_len, arr.shape[1]), pad_val)
                    return np.concatenate([bs, arr], axis=0)
            return arr
            
        return {
            "states": torch.from_numpy(pad(states, 0.0)).float(),
            "actions": torch.from_numpy(pad(actions, 0.0)).float(),
            "returns_to_go": torch.from_numpy(pad(rtg, 0.0)).float().unsqueeze(-1),
            "timesteps": torch.from_numpy(pad(timesteps, 0)).long(),
            "attention_mask": torch.from_numpy(pad(np.ones(real_len), 0.0)).float()
        }

def train_scheduler(
    trajectory_path="trajectories.pkl",
    save_path="./ai_scheduler_model",
    context_length=50,
    epochs=10,
    lr=3e-4,
    batch_size=64,
    rtg_scale=1000.0
):
    print(f"Training AI Scheduler from {trajectory_path}...")
    
    if not os.path.exists(trajectory_path):
        print(f"{trajectory_path} not found.")
        return

    with open(trajectory_path, "rb") as f:
        trajectories = pickle.load(f)
        
    # Preprocessing
    for traj in trajectories:
        rewards = traj['rewards']
        rtg = np.cumsum(rewards[::-1])[::-1]
        traj['rtg'] = rtg.copy()

    trajectories = [t for t in trajectories if len(t['observations']) > 5]
    trajectories.sort(key=lambda x: sum(x['rewards']), reverse=True)
    trajectories = trajectories[:int(len(trajectories)*0.3)] # Top 40%

    stats = get_statistics(trajectories)
    stats['rtg_scale'] = rtg_scale
    trajectories = normalize_trajectories(trajectories, stats, rtg_scale)

    # Save stats
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "normalization_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    dataset = TrajectoryDataset(trajectories, context_length=context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Config (Standard DT for HPO)
    config = DecisionTransformerConfig(
        state_dim=7,
        act_dim=3,
        max_ep_len=1000,
        hidden_size=256,  # 768 -> 256 に下げる
        n_layer=4,        # 12 -> 4 に浅くする
        n_head=4          # 12 -> 4 (Head Dimは64を維持！)
    )
    
    model = DecisionTransformerModel(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(dataloader)*epochs*0.1), num_training_steps=len(dataloader)*epochs)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                states=batch['states'].to(device),
                actions=batch['actions'].to(device),
                returns_to_go=batch['returns_to_go'].to(device),
                timesteps=batch['timesteps'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                return_dict=True
            )
            loss = loss_fn(outputs.action_preds, batch['actions'].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss {total_loss/len(dataloader):.4f}")
        
    model.save_pretrained(save_path)
    print("Training Complete.")

if __name__ == "__main__":
    train_scheduler()
