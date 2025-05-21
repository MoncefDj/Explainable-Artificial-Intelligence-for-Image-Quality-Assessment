# VisEng

## Folder Purpose

This folder contains the visualization and explainability (XAI) pipeline for model predictions. It includes scripts for generating segmentation overlays, saliency maps, and LLM-based textual analysis of results.

## Folder Structure

```
visualize.py     # Functions for plotting images, segmentation, and saliency overlays
llm_analysis.py  # Handles LLM prompt formatting and summary generation
config.py        # All visualization and analysis parameters, paths, and global variables
main.py          # Entry point to run the visualization and LLM analysis pipeline
```

## How to Use

### Prerequisites

- Python 3.8+
- Required packages: `torch`, `torchvision`, `albumentations`, `segmentation-models-pytorch`, `matplotlib`, `pandas`, `opencv-python`, `gpt4all` (for LLM analysis)
- Trained model checkpoint available (path set in `config.py`)
- Test dataset available and paths set in `config.py`
- (Optional) LLM model file for GPT4All, set in `config.py`

### Steps

1. **Edit `config.py`**  
   - Set the model checkpoint path, image/test data paths
   - Adjust visualization and analysis parameters as needed
   - Set LLM model name/path if using LLM analysis

2. **Run Visualization & Analysis**

   Use the provided entry point:

   ```
   python main.py
   ```

   This will generate visualizations and, if configured, LLM-generated summaries.


## Output

- Visualization plots: original image, segmentation overlay, saliency overlay, and analysis summary
- LLM-generated textual summaries (printed to console)
- Ready for further reporting or review