# Train

## Folder Purpose

This folder contains the complete training pipeline for the foreign object segmentation model on chest X-ray images. It provides scripts for data loading, model training, configuration management, and an entry point for running the training process.

## Folder Structure

```
train.py      # Core training logic (data loading, model, training loop)
config.py     # All training parameters, paths, and global variables
main.py       # Entry point to launch the training pipeline
```

## How to Use

### Prerequisites

- Python 3.8+
- Required packages: `torch`, `torchvision`, `albumentations`, `segmentation-models-pytorch`, `tqdm`, `pandas`, `opencv-python`
- Dataset downloaded and all paths set correctly in `config.py`

### Steps

1. **Edit `config.py`**  
   - Set all data paths (CSV files, image directories)
   - Adjust hyperparameters (batch size, epochs, learning rate, etc.) as needed

2. **Run Training**

   Use the provided entry point:

   ```
   python main.py
   ```

   This will automatically use the settings from `config.py`.


## Output

- Model checkpoints for each epoch and the best model (`train_checkpoints/`)
- Training and validation loss history (printed/logged)
- Ready-to-use model weights for downstream testing or deployment