# CXR Image Classification and Explainability (ResNet34 & EfficientNetB0)

This repository contains a **re-implementation** of parts of the work presented in:

- [Explainable Image Quality Assessment for Medical Imaging](https://arxiv.org/abs/2303.14479) by Caner Ozer et al.
- [Explainable Image Quality Analysis of Chest X-Rays (2021)](https://openreview.net/forum?id=ln797A8lAb0) by Caner Ozer and Ilkay Oksuz.

> This project re-implements key components including training and evaluation pipelines, explainability with Grad-CAM and NormGrad, and the Pointing Game metric for localization accuracy.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Directory Structure](#directory-structure)
4. [Dataset Setup](#dataset-setup)
5. [Setup Notes](#setup-notes)
6. [EfficientNetB0 Usage](#efficientnetb0-usage)
7. [ResNet34 Usage](#resnet34-usage)
8. [Explainability Methods](#explainability-methods)
9. [Pointing Game Evaluation](#pointing-game-evaluation)
10. [Citation](#citation)

## Overview

This re-implementation focuses on binary classification of Chest X-Ray (CXR) images into 'affected' or 'normal' using two deep learning models:

- **EfficientNetB0**
- **ResNet34**

Included functionality:
- Model training and evaluation
- Explainability via **Grad-CAM** and **NormGrad**
- **Pointing Game** metric for saliency map evaluation

## Prerequisites

Install dependencies:
```bash
pip install torch torchvision torchaudio numpy scikit-learn matplotlib seaborn pillow tqdm scikit-image pyyaml
```

## Directory Structure

```
/re-implementations
|--- efficientnetb0/
|    |--- train.py
|    |--- test.py
|    |--- efficientnetb0_x_pg.py
|
|--- resnet34/
|    |--- train.py
|    |--- test.py
|    |--- resnet34_x_pg.py
|
|--- models/
|    |--- 
|    |--- 
|
|--- Dataset/
|    |---README.md
|
|--- README.md
```

## Dataset Setup

Data is expected at:
```
|--- object-CXR/
|    |--- train/ (with 'affected' and 'normal' subfolders)
|    |--- dev/   (with 'affected' and 'normal' subfolders)
|    |--- dev.csv
```

The `dev.csv` file must include bounding box annotations in a specific format for use with the Pointing Game.

## Setup Notes

- **Hardcoded Paths:** You may need to update paths in the scripts.
- **Model Naming:** Models are saved with validation accuracy in filenames. Adjust paths or rename files for compatibility with the test and XAI scripts.
- **CUDA Support:** Scripts use GPU if available.

## EfficientNetB0 Usage

### Training
```bash
cd re-implementations/efficientnetb0
python train.py
```

### Testing
```bash
cd re-implementations/efficientnetb0
python test.py
```

### XAI & Pointing Game
```bash
cd re-implementations/efficientnetb0
python efficientnetb0_x_pg.py
```

## ResNet34 Usage

### Training
```bash
cd re-implementations/resnet34
python train.py
```

### Testing
```bash
cd re-implementations/resnet34
python test.py
```

### XAI & Pointing Game
```bash
cd re-implementations/resnet34
python resnet34_x_pg.py
```

## Explainability Methods

Implemented methods:
- **Grad-CAM**: Highlights influential regions of the image.
- **NormGrad**: Computes saliency using gradient-activation norms.
  - Variants: `scaling`, `conv1x1`, `conv3x3`
  - Layers: Final or combined convolutional layers

## Pointing Game Evaluation

Assesses if the saliency map’s peak lies within annotated bounding boxes. Evaluation supports:
- **Single image mode**
- **Batch folder evaluation** (outputs CSV of mean accuracies)

## Citation

If you find this work helpful, please cite the original authors:

```bibtex
@misc{ozer2023explainable,
      title={Explainable Image Quality Assessment for Medical Imaging}, 
      author={Caner Ozer and Arda Guler and Aysel Turkvatan Cansever and Ilkay Oksuz},
      year={2023},
      eprint={2303.14479},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

@inproceedings{Ozer2021,
    title={Explainable Image Quality Analysis of Chest X-Rays},
    author={Ozer, Caner and Oksuz, Ilkay},
    booktitle={Medical Imaging with Deep Learning},
    year={2021},
    url={https://openreview.net/forum?id=ln797A8lAb0}
}
```
