import numpy as np
import pickle
import torch

def get_statistics(trajectories):
    states = np.concatenate([t['observations'] for t in trajectories], axis=0)
    actions = np.concatenate([t['actions'] for t in trajectories], axis=0)
    return {
        'state_mean': np.mean(states, axis=0),
        'state_std': np.std(states, axis=0) + 1e-6,
        'action_mean': np.mean(actions, axis=0),
        'action_std': np.std(actions, axis=0) + 1e-6
    }

def normalize_trajectories(trajectories, stats, rtg_scale):
    for t in trajectories:
        t['observations'] = (t['observations'] - stats['state_mean']) / stats['state_std']
        t['actions'] = (t['actions'] - stats['action_mean']) / stats['action_std']
        # RTG is scaled by constant
        t['rtg'] = t['rtg'] / rtg_scale
    return trajectories

def normalize_state(state, stats):
    """Single state normalization for inference."""
    return (state - stats['state_mean']) / stats['state_std']

def denormalize_action(action, stats):
    """Reverse action normalization for inference."""
    return action * stats['action_std'] + stats['action_mean']

def get_action_values(action):
    """Convert raw action (-1~1) to hyperparameters."""
    # Clip to be safe, though DT output should be reasonable
    action = np.clip(action, -1.0, 1.0)
    
    # LR: -1.0 -> 1e-6, 0.0 -> 3e-5, 1.0 -> 1e-3
    lr_log = action[0] * 1.5 - 4.5
    lr = 10 ** lr_log
    
    weight_decay = (action[1] + 1) / 2 * 0.1
    max_grad_norm = (action[2] + 1) / 2 * 4.9 + 0.1
    
    return {"lr": lr, "weight_decay": weight_decay, "max_grad_norm": max_grad_norm}
