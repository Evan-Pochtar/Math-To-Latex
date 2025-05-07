
import os
import torch
import numpy as np
from trl import GRPOConfig, GRPOTrainer
from rapidfuzz import fuzz
from datasets import Dataset
from datetime import datetime
import pandas as pd

# Create the prompts for GRPO Training
def create_prompt(example):
    example["prompt"] = f"""Please ensure that the following text is valid LaTeX by fixing syntax issues as needed. Here is the potentially invalid LaTeX: {example["prediction"]}. What is the fixed valid LaTeX: """
    return example

# Read in the dataset to use from csv file
csv_path = 'test_predictions.csv'
df = pd.read_csv(csv_path)
df = df.apply(create_prompt, axis=1)
dataset2 = Dataset.from_pandas(df)
print(dataset2)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training will run on: {device}")

# Create a unique checkpoint directory for each run using a timestamp
run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = f'/home/csci5527/shared/the_gradients/5527/{run}'
os.makedirs(checkpoint_dir, exist_ok=True)

def reward(completions, **kwargs):
    """Reward function that rewards a similarity score between two strings in the range [0,1]."""
    correct_latex = kwargs["reference"]
    rewards = []
    for completion, reference in zip(completions, correct_latex):
      if not completion or not reference:
        rewards.append(0.0)
        continue
      # Do not reward empty strings
      if len(completion) == 0:
            rewards.append(0.0)
            continue
      # Perfect match gets a full reward
      if completion == reference:
          rewards.append(1.0)
          continue
      # Apply RapidFuzz ratio for all cases (handles different lengths well)
      similarity = fuzz.ratio(completion, reference) / 100.0
      # # Add additional penalty for length mismatch
      # length_penalty = max(0, 1 - (abs(len(completion) - len(reference)) / max(len(reference), 1)))
      # # Combined score is a linear combination of similarity and length_penalty
      # final_score = (similarity * 0.5) + (length_penalty * 0.5)
      # rewards.append(final_score)
      rewards.append(similarity)
    return rewards

training_args = GRPOConfig(
    output_dir=checkpoint_dir,
    logging_steps=50,
    per_device_train_batch_size=4,  # Decrease this to lower vram usage
    num_generations=4,  # Decrease this to lower vram usage
    save_strategy="no",  # Do not save checkpoints (saves storage space)
    bf16=True,  # Enable bf16 mixed precision on A100 GPUs
)

trainer = GRPOTrainer(
    model="microsoft/Phi-4-mini-instruct",
    reward_funcs=reward,
    args=training_args,
    train_dataset=dataset2,
)

# Train and save the final model
trainer.train()
trainer.save_model(os.path.join(checkpoint_dir, "final_model"))
