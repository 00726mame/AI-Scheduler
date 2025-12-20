from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import torch

class AISchedulerCallback(TrainerCallback):
    """
    A Hugging Face TrainerCallback that uses AIScheduler to dynamically update
    learning rate, weight decay, and max gradient norm.
    """
    def __init__(self, scheduler, verbose=False):
        self.scheduler = scheduler
        self.verbose = verbose
        # Keep track of exponential moving average of loss
        self.ema_loss = None
        self.alpha = 0.9 # Smoothing factor for EMA
        
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called when the Trainer logs information (usually includes loss).
        We use the logged loss to step the scheduler.
        """
        if not state.log_history:
            return
            
        # Get the latest log that has a loss
        last_log = state.log_history[-1]
        
        # Sometimes logs don't have loss (e.g. eval logs), so we look back
        if "loss" not in last_log:
            # Try to find the most recent log with loss
            for log in reversed(state.log_history):
                if "loss" in log:
                    last_log = log
                    break
            else:
                return # No loss found yet
        
        current_loss = last_log["loss"]
        
        # Initialize or update EMA loss
        if self.ema_loss is None:
            self.ema_loss = current_loss
        else:
            self.ema_loss = self.alpha * self.ema_loss + (1 - self.alpha) * current_loss
            
        # Current step info
        current_step = state.global_step
        max_steps = state.max_steps if state.max_steps > 0 else (state.num_train_epochs * 1000) # Estimate if unknown
        
        # Step the scheduler
        try:
            params = self.scheduler.step(
                current_loss=current_loss,
                ema_loss=self.ema_loss,
                current_step=current_step,
                max_steps=max_steps
            )
            
            # Apply parameters to optimizer
            optimizer = kwargs.get('optimizer')
            if optimizer:
                for param_group in optimizer.param_groups:
                    if "lr" in params:
                        param_group["lr"] = params["lr"]
                    if "weight_decay" in params:
                        param_group["weight_decay"] = params["weight_decay"]
                        
                # Note: max_grad_norm is usually handled in TrainingArguments, 
                # but Trainer applies it using args.max_grad_norm.
                # Changing args.max_grad_norm on the fly works in most HF Trainer versions 
                # because it reads args.max_grad_norm inside the training loop step.
                if "max_grad_norm" in params:
                    args.max_grad_norm = params["max_grad_norm"]
                    
            if self.verbose:
                print(f"[AIScheduler] Step {current_step}: Loss={current_loss:.4f} -> LR={params.get('lr'):.2e}")
                
        except Exception as e:
            print(f"[AIScheduler] Warning: Failed to update parameters: {e}")

