# AI Scheduler

AI Scheduler is a **Decision Transformer-based Hyperparameter Scheduler** for LLM training.  
It dynamically adjusts Learning Rate, Weight Decay, and Max Gradient Norm based on training loss and training progress.

## Installation

```bash
pip install transformers torch numpy
```

## Usage

### 1. Basic Usage (Manual Loop)

```python
from ai_scheduler import AIScheduler

# Load Pretrained Scheduler
scheduler = AIScheduler.from_pretrained("./pretrained")

# Inside your training loop
params = scheduler.step(
    current_loss=loss.item(),
    ema_loss=ema_loss,
    current_step=step,
    max_steps=total_steps
)

# Apply params
for param_group in optimizer.param_groups:
    param_group['lr'] = params['lr']
    param_group['weight_decay'] = params['weight_decay']
```

### 2. Hugging Face Trainer Integration

```python
from transformers import Trainer
from ai_scheduler import AIScheduler, AISchedulerCallback

scheduler = AIScheduler.from_pretrained("./pretrained")
callback = AISchedulerCallback(scheduler, verbose=True)

trainer = Trainer(
    ...,
    callbacks=[callback]
)
trainer.train()
```

## Retraining the Scheduler

To train the scheduler on your own trajectory data:

```python
from ai_scheduler.train import train_scheduler

train_scheduler(
    trajectory_path="trajectories.pkl",
    save_path="./my_scheduler_model"
)
```

## Data Collection

You can generate your own training data for the scheduler using a lightweight proxy task (GPT-2 Tiny on TinyStories).
This script employs a **Greedy Search** strategy to explore effective hyperparameter adjustments.

```bash
python -m ai_scheduler.collect --episodes 100 --save_path my_trajectories.pkl
```

After collection, you can retrain the scheduler using the command above.
