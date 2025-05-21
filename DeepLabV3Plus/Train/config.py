import os

# Data paths
DATA_BASE_PATH = "/home/linati/SegmToClass/Train/object-CXR"
TRAIN_CSV = os.path.join(DATA_BASE_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_BASE_PATH, "dev.csv")
TRAIN_IMG_DIR = os.path.join(DATA_BASE_PATH, "train")
TEST_IMG_DIR = os.path.join(DATA_BASE_PATH, "dev")

# Model and training parameters
IMG_SIZE = 256
N_EPOCHS = 50
BATCH_SIZE = 16
N_SPLITS = 5
LEARNING_RATE = 1e-4
NORMGRAD_EPSILON = 0.0005

# Model checkpoint directory
MODEL_SAVE_DIR = "train_checkpoints"

# Kaggle dataset info (optional, for download utility)
KAGGLE_DATASET_NAME = "raddar/foreign-objects-in-chest-xrays"
KAGGLE_DEST_PATH = '/home/linati/SegmToClass/Train'
