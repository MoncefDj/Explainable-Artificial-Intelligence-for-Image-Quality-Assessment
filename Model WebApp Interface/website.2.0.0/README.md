# Intelligent Image Quality Assessment (IQA) Web Application

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask Version](https://img.shields.io/badge/flask-2.x%2B-green.svg)](https://flask.palletsprojects.com/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

![Application Visual Overview](images/visual.png)

This project provides a web-based application for assessing the quality of medical images, specifically Chest X-rays (CXRs), using a Deep Learning-based segmentation model and saliency mapping. It further leverages a Large Language Model (LLM) to provide human-readable explanations of the analysis.

**Authors:** Sadoudi Abdessmad & Djezzar Moncef (2025)

---

## Features

- **AI-Powered Image Analysis:**
  - Utilizes a DeepLabV3+ segmentation model (ResNet50 backbone) to identify potential artifacts or objects in images.
  - Generates NormGrad-based saliency maps to highlight regions influencing model predictions.
  - Calculates quantitative quality scores based on object size, location, and saliency.

- **Interactive Web Interface:**
  - Upload custom images (PNG, JPG, BMP, TIFF) or select from a pre-loaded dataset (if configured).
  - View original, segmentation overlay, and saliency overlay images.
  - Detailed textual analysis report including scoring methodology and parameters.
  - Zoom functionality for all displayed images to inspect details at original resolution.

- **AI Explanation (Optional):**
  - Integrates with a local LLM (e.g., GPT4All with Llama 3) to provide a human-friendly interpretation of the quality assessment results and their potential diagnostic implications.

- **PDF Report Export:**
  - Generate a comprehensive PDF report including quality scores, visual outputs (original, segmentation, saliency at full resolution), AI explanation, and technical analysis.

- **Dark/Light Mode Toggle:** User-friendly interface themes.

- **Ngrok Integration (Optional):** Easily expose the local development server to the internet for testing and demonstration.

- **Modular Architecture:** Codebase structured into services for data loading, model handling, image processing, and report generation for better maintainability and scalability.

---

## Technical Stack

- **Backend:** Python, Flask  
- **Deep Learning:** PyTorch, segmentation-models-pytorch  
- **LLM (Optional):** GPT4All (or other compatible local LLMs)  
- **Image Processing:** OpenCV, Albumentations, Matplotlib, Pillow  
- **Frontend:** HTML, CSS (Bootstrap 5), JavaScript (Vanilla JS, Marked.js, Highlight.js, MathJax)  
- **PDF Generation:** FPDF2  
- **Tunneling (Optional):** pyngrok

---

## Project Structure

```
image_quality_assessment/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_loader_service.py
│   │   ├── model_handler_service.py
│   │   ├── image_processor_service.py
│   │   └── report_generator_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   └── text_utils.py
│   ├── static/
│   │   ├── css/style.css
│   │   ├── js/script.js
│   │   └── images/
│   └── templates/
│       └── index.html
├── Images/
├── config.py
├── run.py
├── uploads/
│   └── temp_originals/
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### 1. Prerequisites

- Python 3.9 or higher
- `pip`
- (Optional) NVIDIA GPU with CUDA drivers

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `config.py`

- `DATA_BASE_PATH`: Path to dataset
- `TRAINED_MODEL_PATH`: Path to `.pth` model file
- `TEST_CSV_NAME`: CSV listing dataset images
- `NGROK_AUTH_TOKEN`: (Optional) For public access
- `LLM Configuration`:
  - `USE_LLM_EXPLANATION`: True or False
  - `LLM_MODEL_NAME`: e.g. `"Meta-Llama-3-8B-Instruct.Q4_0.gguf"`
  - `LLM_DEVICE`: `"cuda"` or `"cpu"`

### 4. (Optional) Download LLM Model

If using `USE_LLM_EXPLANATION = True`, download or run once to auto-download via GPT4All.

### 5. Create Uploads Directory

```bash
mkdir -p uploads/temp_originals
```

Ensure `uploads/` is in `.gitignore`.

---

## Running the Application

```bash
python run.py
```

Then open:

- Local: `http://127.0.0.1:5001`
- Ngrok (if configured): `https://<random>.ngrok-free.app`

---

## Usage

1. Open in browser.
2. Select image:
   - From dataset index
   - Upload manually
3. Toggle AI Explanation (if LLM enabled)
4. Click **Start Analysis**
5. View:
   - Quality scores
   - Visuals (with zoom)
   - Technical + AI reports
6. Export to PDF

---

## License

This project is licensed under the [MIT License](./LICENSE).