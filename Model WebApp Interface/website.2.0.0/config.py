# config.py
import os
import torch

# --- Project Root Path ---
# This assumes config.py is at the root of your project 
# (e.g., image_quality_assessment/config.py)
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# --- Core Paths & Settings ---
DATA_BASE_PATH = os.environ.get("IQAPP_DATA_BASE_PATH", "object-CXR") # DATA SET PATH
TRAINED_MODEL_PATH = os.environ.get("IQAPP_TRAINED_MODEL_PATH", "model.pth") # MODEL PATH
TEST_CSV_NAME = os.environ.get("IQAPP_TEST_CSV_NAME", "dev.csv") # TEST CSV

# UPLOAD_FOLDER will be an absolute path relative to the project root
_UPLOAD_FOLDER_NAME = 'uploads' # Keep the name relative
UPLOAD_FOLDER = os.path.join(APP_ROOT_DIR, _UPLOAD_FOLDER_NAME) 
TEMP_IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, 'temp_originals') 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# --- Analysis Parameters ---
IMG_SIZE = 256
IMG_SIZE_ANALYSIS = IMG_SIZE 

# --- NormGrad Parameters ---
NORMGRAD_EPSILON = 0.0005
NORMGRAD_EPSILON_ANALYSIS = NORMGRAD_EPSILON
NORMGRAD_TARGET_LAYERS_LIST = [
    'encoder.conv1', 'encoder.maxpool', 'encoder.layer1', 'encoder.layer1.0', 'encoder.layer1.0.conv1',
    'encoder.layer1.0.conv2', 'encoder.layer1.0.conv3', 'encoder.layer1.0.downsample', 'encoder.layer1.0.downsample.0',
    'encoder.layer1.0.downsample.1', 'encoder.layer1.1', 'encoder.layer1.1.conv1', 'encoder.layer1.1.conv2',
    'encoder.layer1.1.conv3', 'encoder.layer1.2', 'encoder.layer1.2.conv1', 'encoder.layer1.2.conv2',
    'encoder.layer1.2.conv3', 'encoder.layer2', 'encoder.layer2.0', 'encoder.layer2.0.conv1', 'encoder.layer2.0.conv2',
    'encoder.layer2.0.conv3', 'encoder.layer2.0.downsample', 'encoder.layer2.0.downsample.0', 'encoder.layer2.0.downsample.1',
    'encoder.layer2.1', 'encoder.layer2.1.conv1', 'encoder.layer2.1.conv2', 'encoder.layer2.1.conv3', 'encoder.layer2.2',
    'encoder.layer2.2.conv1', 'encoder.layer2.2.conv2', 'encoder.layer2.2.conv3', 'encoder.layer2.3',
    'encoder.layer2.3.conv1', 'encoder.layer2.3.conv2', 'encoder.layer2.3.conv3', 'encoder.layer3', 'encoder.layer3.0',
    'encoder.layer3.0.conv1', 'encoder.layer3.0.conv2', 'encoder.layer3.0.conv3', 'encoder.layer3.0.downsample',
    'encoder.layer3.0.downsample.0', 'encoder.layer3.0.downsample.1', 'encoder.layer3.1', 'encoder.layer3.1.conv1',
    'encoder.layer3.1.conv2', 'encoder.layer3.1.conv3', 'encoder.layer3.2', 'encoder.layer3.2.conv1',
    'encoder.layer3.2.conv2', 'encoder.layer3.2.conv3', 'encoder.layer3.3', 'encoder.layer3.3.conv1',
    'encoder.layer3.3.conv2', 'encoder.layer3.3.conv3', 'encoder.layer3.4', 'encoder.layer3.4.conv1',
    'encoder.layer3.4.conv2', 'encoder.layer3.4.conv3', 'encoder.layer3.5', 'encoder.layer3.5.conv1',
    'encoder.layer3.5.conv2', 'encoder.layer3.5.conv3', 'encoder.layer4', 'encoder.layer4.0', 'encoder.layer4.0.conv1',
    'encoder.layer4.0.conv2', 'encoder.layer4.0.conv3', 'encoder.layer4.0.downsample', 'encoder.layer4.0.downsample.0',
    'encoder.layer4.0.downsample.1', 'encoder.layer4.1', 'encoder.layer4.1.conv1', 'encoder.layer4.1.conv2',
    'encoder.layer4.1.conv3', 'encoder.layer4.2', 'encoder.layer4.2.conv1', 'encoder.layer4.2.conv2',
    'encoder.layer4.2.conv3', 'decoder', 'decoder.aspp', 'decoder.aspp.0', 'decoder.aspp.0.convs.0',
    'decoder.aspp.0.convs.0.0', 'decoder.aspp.0.convs.0.1', 'decoder.aspp.0.convs.0.2', 'decoder.aspp.0.convs.1',
    'decoder.aspp.0.convs.1.0', 'decoder.aspp.0.convs.1.0.0', 'decoder.aspp.0.convs.1.0.1', 'decoder.aspp.0.convs.1.1',
    'decoder.aspp.0.convs.1.2', 'decoder.aspp.0.convs.2', 'decoder.aspp.0.convs.2.0', 'decoder.aspp.0.convs.2.0.0',
    'decoder.aspp.0.convs.2.0.1', 'decoder.aspp.0.convs.2.1', 'decoder.aspp.0.convs.2.2', 'decoder.aspp.0.convs.3',
    'decoder.aspp.0.convs.3.0', 'decoder.aspp.0.convs.3.0.0', 'decoder.aspp.0.convs.3.0.1', 'decoder.aspp.0.convs.3.1',
    'decoder.aspp.0.convs.3.2', 'decoder.aspp.0.convs.4', 'decoder.aspp.0.convs.4.0', 'decoder.aspp.0.convs.4.1',
    'decoder.aspp.0.convs.4.2', 'decoder.aspp.0.convs.4.3', 'decoder.aspp.0.project', 'decoder.aspp.0.project.0',
    'decoder.aspp.0.project.1', 'decoder.aspp.0.project.2', 'decoder.aspp.0.project.3', 'decoder.aspp.1',
    'decoder.aspp.1.0', 'decoder.aspp.1.1', 'decoder.aspp.2', 'decoder.aspp.3', 'decoder.up', 'decoder.block1',
    'decoder.block1.0', 'decoder.block1.1', 'decoder.block1.2', 'decoder.block2', 'decoder.block2.0',
    'decoder.block2.0.0', 'decoder.block2.0.1', 'decoder.block2.1', 'decoder.block2.2', 'segmentation_head',
    'segmentation_head.0', 'segmentation_head.1', 'segmentation_head.2', 'segmentation_head.2.activation'
]

# --- Scoring Parameters ---
DEFAULT_SIGMOID_K = 15.0
DEFAULT_SIGMOID_THRESH = 0.005

# --- LLM Configuration ---
ANALYSIS_CONFIG_GLOBAL = { 
    "USE_LLM_EXPLANATION": os.environ.get("IQAPP_USE_LLM", "True").lower() == 'true',
    "LLM_MODEL_NAME": os.environ.get("IQAPP_LLM_MODEL_NAME", "Meta-Llama-3-8B-Instruct.Q4_0.gguf"),
    "LLM_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WEIGHT_IMPORTANCE": 0.3, 
    "WEIGHT_SIZE_PENALTY": 0.7,
    "SIGMOID_K_PARAM": 70.0, 
    "SIGMOID_THRESH_PARAM": 0.1, 
    "SALIENCY_FILTER_THRESHOLD": 0.30,
    "NORMGRAD_EPSILON": NORMGRAD_EPSILON_ANALYSIS, 
    "NORMGRAD_ADVERSARIAL": False
}

# --- Ngrok Configuration ---
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "REPLACE_YOUR_TOKEN_HERE") # Ensure this is your actual token

GPT4ALL_AVAILABLE = False
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    print("Config Warning: gpt4all library not found. LLM explanation will be skipped if enabled.")
    if ANALYSIS_CONFIG_GLOBAL["USE_LLM_EXPLANATION"]:
        ANALYSIS_CONFIG_GLOBAL["USE_LLM_EXPLANATION"] = False 