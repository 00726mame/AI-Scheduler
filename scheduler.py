import torch
import numpy as np
import pickle
import os
from transformers import DecisionTransformerModel
try:
    from .utils import normalize_state, denormalize_action, get_action_values
except ImportError:
    from utils import normalize_state, denormalize_action, get_action_values

class AIScheduler:
    def __init__(self, model_path, context_length=50, device=None, stats_path=None, adaptive=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.context_length = context_length
        self.adaptive = adaptive
        
        # Load Stats
        if stats_path is None:
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
        
        # Adaptive Stats Tracking
        # We track running mean/std for observation features: loss, ema_loss, grad_norm, weight_norm, delta_loss
        # Indices in state vector: [0, 1, 2, 3, 6]
        self.running_mean = np.zeros(7, dtype=np.float32)
        self.running_var = np.ones(7, dtype=np.float32)
        self.count = 1e-4
        
        # Initialize running stats with trained stats to avoid cold start shock, 
        # but allow rapid adaptation (low initial count effectively)
        self.running_mean = self.stats['state_mean'].copy()
        # self.running_var = self.stats['state_std'].copy() ** 2 
        # Actually, let's start fresh for critical metrics if adaptive is on, 
        # but defaulting to training stats is safer as a prior.
        
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
        self.rtg_current = 100.0 
        
        # State tracking
        self.prev_loss = None
        # Initialize action (corresponding to neutral/zero hyperparameters)
        self.current_action_raw = np.zeros(3, dtype=np.float32)
        
    def update_stats(self, x):
        """Update running mean and variance using Welford's online algorithm."""
        self.count += 1
        delta = x - self.running_mean
        self.running_mean += delta / self.count
        delta2 = x - self.running_mean
        self.running_var += delta * delta2

    def step(self, current_loss, ema_loss, grad_norm, weight_norm, current_step, max_steps):
        """
        Step the scheduler to get new hyperparameters.
        
        Args:
            current_loss (float): Current training loss.
            ema_loss (float): Exponential moving average of loss.
            grad_norm (float): Gradient norm.
            weight_norm (float): Weight norm.
            current_step (int): Current step in the episode/epoch.
            max_steps (int): Total steps in the episode/epoch.
            
        Returns:
            dict: New hyperparameters {lr, weight_decay, max_grad_norm}
        """
        # Handle first step
        if self.prev_loss is None:
            self.prev_loss = current_loss

        # Calculate derived state features
        delta_loss = current_loss - self.prev_loss
        
        # Calculate Log LR from current action
        # Note: We use the action from the PREVIOUS step (which generated the current loss)
        params = get_action_values(self.current_action_raw)
        current_log_lr = np.log10(params["lr"] + 1e-9)

        # 1. Construct State
        # State: [loss, ema_loss, grad_norm, weight_norm, progress, log_lr, delta_loss]
        progress = current_step / max_steps
        raw_state = np.array([
            current_loss, 
            ema_loss, 
            grad_norm, 
            weight_norm, 
            progress, 
            current_log_lr, 
            delta_loss
        ], dtype=np.float32)
        
        # Update Adaptive Stats
        if self.adaptive:
            # Only adapt: loss(0), ema_loss(1), grad_norm(2), weight_norm(3), delta_loss(6)
            # Keep progress(4) and log_lr(5) as they are standard/controlled
            mask = np.array([1, 1, 1, 1, 0, 0, 1], dtype=bool)
            
            # Welford update
            self.count += 1
            delta = raw_state - self.running_mean
            self.running_mean[mask] += delta[mask] / self.count
            delta2 = raw_state - self.running_mean
            self.running_var[mask] += delta[mask] * delta2[mask]
        
        # Update prev_loss for next step
        self.prev_loss = current_loss
        
        # Normalize
        if self.adaptive:
            std = np.sqrt(self.running_var / (self.count + 1e-6)) + 1e-6
            # Use training mean/std for non-adapted indices
            mean_to_use = self.running_mean.copy()
            std_to_use = std.copy()
            
            # Override non-adapted with original stats (progress, log_lr)
            # Actually, log_lr should likely match training distribution?
            # If we output LR, we want it to be interpreted correctly.
            mean_to_use[4] = self.stats['state_mean'][4] # Progress
            std_to_use[4] = self.stats['state_std'][4]
            mean_to_use[5] = self.stats['state_mean'][5] # Log LR
            std_to_use[5] = self.stats['state_std'][5]
            
            norm_state = (raw_state - mean_to_use) / std_to_use
        else:
            norm_state = normalize_state(raw_state, self.stats)
            
        state_tensor = torch.from_numpy(norm_state).float().to(self.device).reshape(1, 1, -1)
        
        # 2. Update Context
        # Shift buffers if full
        if self.current_t >= self.context_length:
            self.states = torch.roll(self.states, -1, dims=1)
            self.actions = torch.roll(self.actions, -1, dims=1)
            self.rewards = torch.roll(self.rewards, -1, dims=1) 
            self.timesteps = torch.roll(self.timesteps, -1, dims=1)
            self.attention_mask = torch.roll(self.attention_mask, -1, dims=1)
            pos = self.context_length - 1
        else:
            pos = self.current_t
            
        self.states[0, pos] = state_tensor
        self.timesteps[0, pos] = current_step
        self.attention_mask[0, pos] = 1.0
        
        # RTG Management
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
        
        # Update internal tracking
        self.current_action_raw = real_action_values
        
        # Store action for next step context
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

