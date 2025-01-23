

# Open Implemenation of Deepseek's R1

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


## Installation

```bash
pip install openr1
```

## Usage

```python
"""
This example demonstrates a complete pipeline for training a language model using GRPO.
It includes a basic reward model and a full training loop implementation.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from openr1.main import GRPO, GRPOConfig


class SimpleRewardModel:
    """
    A basic reward model that evaluates text quality using a pretrained classifier.
    In practice, this would be replaced with more sophisticated reward mechanisms
    like human feedback, task-specific metrics, or a dedicated reward model.
    """
    def __init__(self, model_name: str = "facebook/opt-350m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1  # Single score output
        ).to(self.device)
    
    def compute_rewards(self, texts: List[str]) -> torch.Tensor:
        """
        Compute reward scores for generated texts.
        Returns a tensor of rewards in the range [0, 1].
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            rewards = torch.sigmoid(outputs.logits).squeeze(-1)
        
        return rewards.cpu()

class PromptDataset(Dataset):
    """
    Dataset for training prompts. This simple implementation just wraps
    a list of prompts, but could be extended to include additional metadata,
    task information, or structured input-output pairs.
    """
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]

class TrainingLogger:
    """
    Simple training logger that saves metrics to disk and prints updates.
    Maintains running statistics for easy progress monitoring.
    """
    def __init__(self, log_dir: str = "training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.metrics_history = []
        self.running_stats = {}
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a single training step."""
        metrics['step'] = step
        self.metrics_history.append(metrics)
        
        # Update running statistics
        for key, value in metrics.items():
            if key not in self.running_stats:
                self.running_stats[key] = []
            self.running_stats[key].append(value)
            if len(self.running_stats[key]) > 100:  # Keep last 100 values
                self.running_stats[key].pop(0)
    
    def get_running_averages(self) -> Dict[str, float]:
        """Get running averages of all metrics."""
        return {
            k: np.mean(v) for k, v in self.running_stats.items()
            if k != 'step'
        }
    
    def save_logs(self):
        """Save all metrics to disk."""
        with open(self.log_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

def train_step(
    grpo: GRPO,
    prompts: List[str],
    reward_model: SimpleRewardModel,
    step: int
) -> Dict[str, float]:
    """
    Perform a single training step with proper sequence length handling.
    """
    # Set proper padding in tokenizer
    grpo.tokenizer.padding_side = 'left'  # Important for decoder-only models
    grpo.tokenizer.pad_token = grpo.tokenizer.eos_token  # Ensure pad token is set
    
    # Generate responses with explicit max length
    generations, logits = grpo.generate(
        prompts,
        num_samples=grpo.config.group_size,
        max_length=grpo.config.max_sequence_length,
        pad_token_id=grpo.tokenizer.pad_token_id
    )
    
    # Flatten generations for reward computation
    flat_generations = [text for sublist in generations for text in sublist]
    
    # Compute rewards
    rewards = reward_model.compute_rewards(flat_generations)
    
    # Create group indices
    group_indices = torch.arange(len(flat_generations)) // grpo.config.group_size
    
    # Tokenize with careful length handling
    encoded = grpo.tokenizer(
        flat_generations,
        padding=True,
        truncation=True,
        max_length=grpo.config.max_sequence_length,
        return_tensors="pt",
        return_attention_mask=True
    )
    
    # Ensure logits and input_ids have matching sequence lengths
    max_len = min(encoded.input_ids.size(1), logits.size(1))
    input_ids = encoded.input_ids[:, -max_len:]
    attention_mask = encoded.attention_mask[:, -max_len:]
    logits = logits[:, :max_len]
    
    # Update policy with aligned tensors
    metrics = grpo.update(
        input_ids=input_ids,
        attention_mask=attention_mask,
        rewards=rewards,
        group_indices=group_indices,
        old_logits=logits
    )
    
    metrics.update({
        "mean_reward": rewards.mean().item(),
        "max_reward": rewards.max().item(),
        "min_reward": rewards.min().item(),
        "sequence_length": max_len,
    })
    
    return metrics

def main():
    # Initialize models and tokenizer
    print("Initializing models...")
    model_name = "facebook/opt-125m"  # Using a small model for example purposes
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize GRPO with custom configuration
    grpo = GRPO(
        model=model,
        tokenizer=tokenizer,
        config=GRPOConfig(
            group_size=4,  # Number of generations per prompt
            learning_rate=1e-5,
            max_sequence_length=128
        )
    )
    
    # Initialize reward model and logger
    reward_model = SimpleRewardModel()
    logger = TrainingLogger()
    
    # Example prompts for training - in practice, you'd load these from a file
    prompts = [
        "Write a clear explanation of how photosynthesis works:",
        "Describe the process of making bread from scratch:",
        "Explain the concept of gravity to a child:",
        "Write a story about a robot discovering emotions:",
    ]
    
    # Create dataset and dataloader
    dataset = PromptDataset(prompts)
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Small batch size for example
        shuffle=True
    )
    
    # Training loop
    print("Starting training...")
    num_epochs = 3
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Progress bar for each epoch
        pbar = tqdm(dataloader)
        for batch_prompts in pbar:
            # Perform training step
            metrics = train_step(grpo, batch_prompts, reward_model, global_step)
            logger.log_metrics(metrics, global_step)
            
            # Update progress bar with current metrics
            running_avgs = logger.get_running_averages()
            pbar.set_postfix({
                'loss': f"{running_avgs['total_loss']:.4f}",
                'reward': f"{running_avgs['mean_reward']:.4f}"
            })
            
            global_step += 1
        
        # Generate example completions at the end of each epoch
        test_prompt = "Explain the importance of exercise:"
        print(f"\nExample generations for: {test_prompt}")
        generations, _ = grpo.generate([test_prompt], num_samples=2)
        for i, gen in enumerate(generations[0], 1):
            print(f"\nGeneration {i}:")
            print(gen)
    
    # Save final model and logs
    print("\nSaving model and logs...")
    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")
    logger.save_logs()
    print("Training complete!")

if __name__ == "__main__":
    main()
```


## Diagram


```mermaid
sequenceDiagram
    participant Input as Input Data
    participant Policy as Policy Network
    participant Group as Group Processing
    participant Loss as Loss Computation
    participant Optim as Optimizer
    
    Input->>Policy: Forward Pass
    Policy->>Group: Generate Group Samples
    Group->>Group: Compute Statistics
    Group->>Loss: Calculate Advantages
    Policy->>Loss: Compute KL Divergence
    Loss->>Loss: Compute Total Loss
    Loss->>Optim: Backward Pass
    Optim->>Policy: Update Parameters
```