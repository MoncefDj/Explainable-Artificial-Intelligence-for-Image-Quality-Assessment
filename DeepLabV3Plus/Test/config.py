import os

# Data paths
DATA_BASE_PATH = "/home/linati/SegmToClass/Train/object-CXR"
TEST_CSV = os.path.join(DATA_BASE_PATH, "dev.csv")
TEST_IMG_DIR = os.path.join(DATA_BASE_PATH, "dev")

# Model checkpoint path (update as needed)
MODEL_CHECKPOINT_PATH = "../Train/train_checkpoints/best_model.pth"

# Inference parameters
IMG_SIZE = 256
BATCH_SIZE = 16
THRESHOLD = 0.5
