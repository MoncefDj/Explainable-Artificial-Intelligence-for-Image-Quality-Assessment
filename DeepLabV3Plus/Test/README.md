# Test

## Folder Purpose

This folder provides the testing and inference pipeline for the trained segmentation model. It enables users to load a trained model, run predictions on test images, and evaluate or save the results.

## Folder Structure

```
test.py      # Core testing/inference logic (model loading, prediction)
config.py    # All testing parameters, paths, and global variables
main.py      # Entry point to launch the testing pipeline
```

## How to Use

### Prerequisites

- Python 3.8+
- Required packages: `torch`, `torchvision`, `albumentations`, `segmentation-models-pytorch`, `pandas`, `opencv-python`
- Trained model checkpoint available (path set in `config.py`)
- Test dataset available and paths set in `config.py`

### Steps

1. **Edit `config.py`**  
   - Set the path to the trained model checkpoint
   - Set test CSV and image directory paths
   - Adjust batch size, image size, and threshold as needed

2. **Run Testing**

   Use the provided entry point:

   ```
   python main.py
   ```

   This will automatically use the settings from `config.py`.


## Output

- Model predictions on the test set (can be saved or further processed)
- Optionally, evaluation metrics or prediction files (modify `main.py` as needed)
- Console output with progress and basic results