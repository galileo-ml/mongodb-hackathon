"""Configuration for Fireworks.ai fine-tuning."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fireworks API Configuration
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Base model to fine-tune
# Options: "accounts/fireworks/models/llama-v3p1-8b-instruct"
#          "accounts/fireworks/models/qwen2p5-7b-instruct"
#          "accounts/fireworks/models/gemma-2-9b-it"
BASE_MODEL = os.getenv("BASE_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct")

# Fine-tuning parameters
LORA_RANK = int(os.getenv("LORA_RANK", "8"))  # Must be power of 2, max 64
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))  # Default learning rate
EPOCHS = int(os.getenv("EPOCHS", "3"))  # Number of training epochs
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))  # Batch size for training

# Dataset paths
TRAINING_DATA_PATH = "data/context_aggregation_training.jsonl"

# Model naming
FINETUNE_JOB_NAME = os.getenv("FINETUNE_JOB_NAME", "context-aggregation-model-v1")

# Validation
if not FIREWORKS_API_KEY:
    raise ValueError(
        "FIREWORKS_API_KEY not found in environment variables. "
        "Please create a .env file with your API key."
    )
