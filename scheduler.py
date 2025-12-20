import torch
import numpy as np
import pickle
import os
from transformers import DecisionTransformerModel
from .utils import normalize_state, denormalize_action, get_action_values

class AIScheduler:
    def __init__(self, model_path, context_length=20, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length
        
        # Load Stats
        stats_path = os.path.join(model_path, "normalization_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found at {stats_path}")
            
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
            
        self.rtg_scale = self.stats.get('rtg_scale', 1000.0)
        
        # Load Model
        self.model = DecisionTransformerModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # History Buffer
        self.reset()
        
    def reset(self):
        """Reset the history buffer for a new episode."""
        self.states = torch.zeros((1, self.context_length, self.model.config.state_dim), device=self.device)
        self.actions = torch.zeros((1, self.context_length, self.model.config.act_dim), device=self.device)
        self.rewards = torch.zeros((1, self.context_length, 1), device=self.device)
        self.timesteps = torch.zeros((1, self.context_length), dtype=torch.long, device=self.device)
        self.attention_mask = torch.zeros((1, self.context_length), device=self.device)
        
        self.current_t = 0
        self.target_return = 0.0 # Initial target return? Or cumulative? DT usually uses RTG.
        # In the training script, RTG was calculated as cumulative sum of future rewards.
        # For inference, we need to specify a Desired Return (RTG).
        # However, the user request scneario seems to be "Scheduler", where we might want to *maximize* reward.
        # But wait, `2_train_dt.py` training logic:
        # rtg = np.cumsum(rewards[::-1])[::-1]
        # And reward was based on Loss decrease.
        # So providing a high RTG means "I want the loss to decrease a lot".
        
        # Let's set a default high initial RTG if not specified, 
        # or update it dynamically? 
        # Standard DT inference: params are (states, actions, returns_to_go, timesteps)
        # We need to maintain the sequence.
        
        # For this specific scheduler, let's keep it simple: 
        # The user just wants the next action.
        
        # For inference stability, we usually pick a target return (e.g. max realized in training * 1.2)
        # I will expose a method to set target return.
        self.rtg_current = 100.0 # Default high value to encourage good performance
        
    def step(self, current_loss, ema_loss, current_step, max_steps):
        """
        Step the scheduler to get new hyperparameters.
        
        Args:
            current_loss (float): Current training loss.
            ema_loss (float): Exponential moving average of loss.
            current_step (int): Current step in the episode/epoch.
            max_steps (int): Total steps in the episode/epoch.
            
        Returns:
            dict: New hyperparameters {lr, weight_decay, max_grad_norm}
        """
        # 1. Construct State
        # State: [loss, ema_loss, 0, 0, progress]
        progress = current_step / max_steps
        raw_state = np.array([current_loss, ema_loss, 0.0, 0.0, progress], dtype=np.float32)
        
        # Normalize
        norm_state = normalize_state(raw_state, self.stats)
        state_tensor = torch.from_numpy(norm_state).float().to(self.device).reshape(1, 1, -1)
        
        # 2. Update Context
        # Shift buffers if full
        if self.current_t >= self.context_length:
            self.states = torch.roll(self.states, -1, dims=1)
            self.actions = torch.roll(self.actions, -1, dims=1)
            self.rewards = torch.roll(self.rewards, -1, dims=1) # RTG effectively
            self.timesteps = torch.roll(self.timesteps, -1, dims=1)
            self.attention_mask = torch.roll(self.attention_mask, -1, dims=1) # Keep full mask
            pos = self.context_length - 1
        else:
            pos = self.current_t
            
        self.states[0, pos] = state_tensor
        self.timesteps[0, pos] = current_step
        self.attention_mask[0, pos] = 1.0
        
        # RTG Management
        # In standard DT inference, we feed (Initial RTG - reward_so_far).
        # But here we treat it as a "prompt" for "High Return Behavior".
        # We just feed a constant high RTG normalized.
        # Or better: construct RTG input based on self.rtg_current
        # Note: The model expects Normalized RTG.
        rtg_val = self.rtg_current / self.rtg_scale
        self.rewards[0, pos] = rtg_val 
        
        # 3. Inference
        with torch.no_grad():
            outputs = self.model(
                states=self.states,
                actions=self.actions,
                returns_to_go=self.rewards,
                timesteps=self.timesteps,
                attention_mask=self.attention_mask,
                return_dict=True
            )
            action_preds = outputs.action_preds
            
        # Get last action
        raw_action = action_preds[0, pos].cpu().numpy()
        
        # 4. Denormalize & process
        real_action_values = denormalize_action(raw_action, self.stats)
        
        # Update action buffer for next step auto-regressiveness
        # Note: In true autoregression we feed the *actual* action taken. 
        # If the user overrides, this might diverge, but we assume they use what we suggest.
        # We should store the *normalized* predicted action back into buffer?
        # Standard DT: uses the action that was actually taken.
        # Ideally step() should accept `prev_action` but for first step it's zeros.
        # Let's simplify and assume the predicted action is taken.
        # Make sure to put it in the buffer for the *next* step logic, but wait, 
        # DT predicts action a_t given s_t. We need to store a_t for s_{t+1}.
        # So we store it at `pos`.
        self.actions[0, pos] = torch.from_numpy(raw_action).to(self.device) # Store the RAW (normalized output) action
        
        self.current_t += 1
        
        # Convert to HPO params
        hpo_params = get_action_values(real_action_values)
        return hpo_params

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        """
        Load a pretrained AIScheduler.
        
        Args:
            path (str): Path to directory containing model.safetensors and normalization_stats.pkl
        """
        return cls(path, **kwargs)

