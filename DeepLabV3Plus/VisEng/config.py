import os

# Paths
DATA_BASE_PATH = "/home/linati/SegmToClass/Train/object-CXR"
TRAIN_CSV = os.path.join(DATA_BASE_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_BASE_PATH, "dev.csv")
TRAIN_IMG_DIR = os.path.join(DATA_BASE_PATH, "train")
TEST_IMG_DIR = os.path.join(DATA_BASE_PATH, "dev")
MODEL_PATH = "/path/to/best_model.pth"  # Update this to your trained model checkpoint

# Visualization/Analysis parameters
IMG_SIZE = 256
BATCH_SIZE = 16
NORMGRAD_EPSILON = 0.0005
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

# LLM configuration
LLM_MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
LLM_MAX_TOKENS = 8192
LLM_TEMP = 0.1

# Analysis/Scoring parameters
SIGMOID_K = 70.0
SIGMOID_THRESH = 0.1
WEIGHT_IMPORTANCE = 0.3
WEIGHT_SIZE_PENALTY = 0.7
SALIENCY_FILTER_THRESHOLD = 0.30

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
