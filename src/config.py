import os

import torch

# Hyperparameters
RANDOM_SEED = 42
LEARNING_RATE = 0.005
BATCH_SIZE = 128
NUM_EPOCHS = 10

# Architecture
NUM_FEATURES = 32 * 32
NUM_CLASSES = 10

# Other
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = os.cpu_count()
