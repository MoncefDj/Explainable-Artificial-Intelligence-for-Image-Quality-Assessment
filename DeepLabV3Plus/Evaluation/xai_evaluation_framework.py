'''
Explainable AI (XAI) Faithfulness and Localization Metrics Evaluation Framework.

This script provides a comprehensive framework for evaluating XAI methods 
applied to image segmentation models, particularly focusing on faithfulness and 
localization metrics. It includes:
1.  A k-fold model evaluation pipeline that loads pre-trained model checkpoints 
    and computes various XAI metrics (DAUC, IAUC, Drop in Performance, 
    Pointing Game Accuracy) using combined saliency maps from multiple layers.
2.  Implementations of XAI techniques like Grad-CAM and NormGrad (Order 0, Order 1).
3.  Helper functions for metric calculation, image perturbation, and data loading.
4.  An additional example function (`run_faithfulness_localization_example_v4`) 
    demonstrating a specific use case of the evaluation metrics.

The script is configured to log its operations and save results to a CSV file.
'''

import os
import glob
import tempfile
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from sklearn.metrics import auc as sklearn_auc
from tqdm import tqdm
import copy
import logging
import time
import sys
import io
import albumentations as A # For test_transform
from segmentation_models_pytorch import DeepLabV3Plus # For SegmentationModel

# --- Logger Setup --- (From Block 3 of the provided snippet)
LOG_FILE_PATH = 'evaluation_kfold_models_log.txt'
RESULTS_CSV_PATH = 'kfold_evaluation_metrics_results.csv'

# Remove old log file if it exists
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console Handler (optional, for real-time feedback if not running in a pure script)
# console_handler = logging.StreamHandler(sys.stdout)
# console_formatter = logging.Formatter('%(asctime)s - %(message)s')
# console_handler.setFormatter(console_formatter)
# logger.addHandler(console_handler) # Uncomment if you want console output as well

# Redirect stdout and stderr to logger
class LoggerWriter:
    def __init__(self, log_func):
        self.log_func = log_func
        self.buffer = []

    def write(self, message):
        if message.strip() != "":  # Avoid logging empty lines
            self.log_func(message.strip())

    def flush(self):
        pass # Handled by logger

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

logger.info("--- XAI Evaluation Framework Script Started ---")

# --- Configuration & Global Variables (from Block 3, for k-fold) ---
DATA_BASE_PATH_EXPECTED = "/content/object-CXR" # Example, will be overridden by BASE_DATA_PATH
IMG_SIZE = globals().get('IMG_SIZE', 256)
N_EPOCHS = globals().get('N_EPOCHS', 5) # Not directly used in this eval script
BATCH_SIZE = globals().get('BATCH_SIZE', 16)
N_SPLITS = globals().get('N_SPLITS', 5) # Not directly used
LEARNING_RATE = globals().get('LEARNING_RATE', 1e-4) # Not directly used
NORMGRAD_EPSILON = globals().get('NORMGRAD_EPSILON', 0.0005)

BASE_DATA_PATH = "/home/linati/object-CXR_EB0/object-CXR" # USER: Verify this path
DEV_CSV_FILENAME = "dev.csv"
DEV_IMG_DIR_NAME = "dev"
AFFECTED_IMAGE_SUBDIR = "affected"

affected_img_dir_path = os.path.join(BASE_DATA_PATH, DEV_IMG_DIR_NAME, AFFECTED_IMAGE_SUBDIR)
dev_csv_path = os.path.join(BASE_DATA_PATH, DEV_CSV_FILENAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

LIST_OF_LAYERS_TO_COMBINE_KFOLD = [
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
NUM_AFFECTED_SAMPLES_TO_USE_KFOLD = 500
RANDOM_SEED_FOR_SAMPLING_KFOLD = 42
NUM_BATCHES_FOR_XAI_EVAL_KFOLD = None
AUC_STEPS_KFOLD = 10
DROP_TOP_K_KFOLD = 0.2
NORMGRAD_EPSILON_FOR_EVAL_KFOLD = NORMGRAD_EPSILON

MODEL_BASE_DIR_KFOLD = "/home/linati/SegmToClass/Train/train_checkpoints/run_20250508_205623/models" # USER: Verify this path

logger.info(f"Target model directory for k-fold evaluation: {MODEL_BASE_DIR_KFOLD}")
logger.info(f"Number of layers for k-fold combined heatmap: {len(LIST_OF_LAYERS_TO_COMBINE_KFOLD)}")
logger.info(f"Number of affected samples for k-fold: {NUM_AFFECTED_SAMPLES_TO_USE_KFOLD}")

# --- Core Helper Functions (definitions from Block 3) ---

def parse_annotations(ann_str, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if not isinstance(ann_str, str) or ann_str.strip() == "":
        return mask
    h, w = img_shape[:2]
    try:
        objects = ann_str.split(';')
        for obj in objects:
            if not obj.strip(): continue
            parts = list(map(int, obj.split()))
            obj_type = parts[0]; coords = parts[1:]
            if obj_type == 0 and len(coords) == 4: # Rectangle
                x1,y1,x2,y2 = map(lambda v:max(0,v), coords); x2,y2=min(x2,w),min(y2,h)
                if x1<x2 and y1<y2: mask[y1:y2,x1:x2]=1
            elif obj_type == 1 and len(coords) == 4: # Ellipse
                x1,y1,x2,y2 = map(lambda v:max(0,v), coords); x2,y2=min(x2,w),min(y2,h)
                if x1<x2 and y1<y2:
                    center=((x1+x2)//2,(y1+y2)//2); axes=((x2-x1)//2,(y2-y1)//2)
                    if axes[0]>0 and axes[1]>0: cv2.ellipse(mask,center,axes,0,0,360,1,-1)
            elif obj_type == 2 and len(coords) >= 6 and len(coords) % 2 == 0: # Polygon
                points = np.array(coords).reshape(-1,2)
                points[:,0]=np.clip(points[:,0],0,w-1); points[:,1]=np.clip(points[:,1],0,h-1)
                cv2.fillPoly(mask,[points],1)
    except Exception as e:
        logger.error(f"Error parsing annotation string '{ann_str}': {e}")
    return mask

class ObjectCXRSataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, img_size=(256, 256)):
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            logger.error(f"Error: CSV file not found at {csv_file}")
            self.data = pd.DataFrame(); return
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        if idx >= len(self.data): raise IndexError("Index out of bounds")
        img_name = self.data.iloc[idx, 0]
        ann_str = self.data.iloc[idx, 1] if self.data.shape[1] > 1 else "" # Handle missing annotation col
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not load image {img_path}. Returning dummy data.")
                return torch.zeros((3, *self.img_size)), torch.zeros((1, *self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_shape = img.shape
            mask = parse_annotations(ann_str, original_shape)
            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
            else:
                img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            if img.dtype == np.uint8: img = img / 255.0
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return img, mask
        except Exception as e:
            logger.error(f"Error processing item at index {idx} (image: {img_name}): {e}")
            logger.error(traceback.format_exc())
            return torch.zeros((3, *self.img_size)), torch.zeros((1, *self.img_size))

test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

class SegmentationModel(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", classes=1):
        super().__init__()
        self.model = DeepLabV3Plus(
            encoder_name=encoder_name, encoder_weights=encoder_weights,
            classes=classes, activation=None
        )
    def forward(self, x): return self.model(x)

def find_layer(model, layer_name):
    current_module = model
    try:
        for part in layer_name.split('.'):
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
            else:
                try:
                    idx = int(part)
                    if isinstance(current_module, (nn.Sequential, nn.ModuleList)):
                        current_module = current_module[idx]
                    else: raise AttributeError(f"Module {type(current_module)} not Seq/List")
                except (ValueError, IndexError, AttributeError, TypeError): return None
        return current_module
    except Exception: return None

def gradcam_segmentation_data_intermediate(model, img_tensor, device, target_layer_module):
    if target_layer_module is None: logger.error("Grad-CAM: target_layer_module is None."); return None
    model.eval(); activations = []; gradients = []
    def fwd_hook(mod, inp, outp): act=outp[0] if isinstance(outp,(list,tuple)) else outp; activations.append(act.detach())
    def bwd_hook(mod, grad_i, grad_o):
        if grad_o[0] is not None: gradients.append(grad_o[0].detach())
        else: logger.warning(f"Grad-CAM: backward hook None grad for {target_layer_module}."); gradients.append(None)
    h_fwd = target_layer_module.register_forward_hook(fwd_hook)
    h_bwd = target_layer_module.register_full_backward_hook(bwd_hook)
    img_clone = img_tensor.clone().detach().to(device).requires_grad_(True)
    model.zero_grad(); cam_resized = None
    try:
        output = model(img_clone)
        if output.nelement() == 0: logger.warning("Grad-CAM: Model output empty."); return None
        score = torch.sigmoid(output).mean(); score.backward()
        if not activations: logger.warning("Grad-CAM: No activations."); return None
        if not gradients or gradients[0] is None: logger.warning("Grad-CAM: No gradients."); return None
        act = activations[-1]; grad = gradients[-1]
        pooled_grads = grad.mean(dim=[0,2,3], keepdim=True)
        cam = (act * pooled_grads).sum(dim=1); cam = F.relu(cam.squeeze(0))
        cam_np = cam.cpu().numpy()
        if np.max(cam_np) > 1e-8: cam_np = cam_np / np.max(cam_np)
        else: cam_np = np.zeros_like(cam_np)
        img_hw = img_tensor.shape[2:]
        cam_resized = cv2.resize(cam_np, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_LINEAR)
    except RuntimeError as e: logger.error(f"Grad-CAM Error: {e}"); return None
    finally:
        h_fwd.remove(); h_bwd.remove(); model.zero_grad()
        if img_clone.grad is not None: img_clone.grad.zero_()
    return cam_resized

def compute_normgrad_order0(model, input_tensor, target_layer_module, device):
    if target_layer_module is None: logger.error("NormGrad0: target_layer_module is None."); return None
    model.eval(); model.zero_grad(); activations = None; gradients = None
    def fwd_hook(mod, inp, outp): nonlocal activations; act=outp[0] if isinstance(outp,(list,tuple)) else outp; activations=act.detach()
    def bwd_hook(mod, grad_i, grad_o):
        nonlocal gradients
        if grad_o[0] is not None: gradients = grad_o[0].detach()
        else: logger.warning(f"NormGrad0: backward hook None grad for {target_layer_module}."); gradients = None
    h_fwd = target_layer_module.register_forward_hook(fwd_hook)
    h_bwd = target_layer_module.register_full_backward_hook(bwd_hook)
    input_clone = input_tensor.clone().detach().to(device).requires_grad_(True)
    normgrad_map_np = None
    try:
        output = model(input_clone); score = torch.sigmoid(output).mean()
        model.zero_grad(); score.backward()
        if activations is None or gradients is None: logger.error(f"NormGrad0: Failed acts/grads for {target_layer_module}."); return None
        norm_acts = torch.linalg.norm(activations, ord=2, dim=1, keepdim=False)
        norm_grads = torch.linalg.norm(gradients, ord=2, dim=1, keepdim=False)
        normgrad_map = norm_acts * norm_grads
        normgrad_map_np = normgrad_map.squeeze(0).cpu().numpy()
    except Exception as e: logger.error(f"NormGrad0 Error: {e}"); return None
    finally:
        h_fwd.remove(); h_bwd.remove(); model.zero_grad()
        if input_clone.grad is not None: input_clone.grad.zero_()
    return normgrad_map_np

def compute_normgrad_order1_by_name(original_model, input_tensor, target_layer_name, device, epsilon=0.0005, adversarial=False):
    model_for_grad = copy.deepcopy(original_model).to(device); model_for_grad.eval(); model_for_grad.zero_grad()
    input_clone_grad = input_tensor.clone().detach().to(device).requires_grad_(True)
    try:
        output_orig = model_for_grad(input_clone_grad); score_orig = torch.sigmoid(output_orig).mean(); score_orig.backward()
    except Exception as e: logger.error(f"NormGrad1 param grad Error: {e}"); del model_for_grad; return None
    param_grads = {name: param.grad.data.clone().detach() for name, param in model_for_grad.named_parameters() if param.grad is not None}
    del model_for_grad
    if not param_grads: logger.warning("NormGrad1: No param grads for update.")
    model_prime = copy.deepcopy(original_model).to(device); model_prime.eval()
    with torch.no_grad():
        for name, param in model_prime.named_parameters():
            if name in param_grads:
                alpha = epsilon if adversarial else -epsilon
                param.add_(param_grads[name], alpha=alpha)
    prime_base_model = model_prime.model if hasattr(model_prime,'model') else model_prime
    prime_target_module = find_layer(prime_base_model, target_layer_name)
    if prime_target_module is None: logger.error(f"NormGrad1: Cannot find '{target_layer_name}' in prime model."); del model_prime; return None
    input_clone_order0 = input_tensor.clone().detach().to(device)
    normgrad_map_np = compute_normgrad_order0(model_prime, input_clone_order0, prime_target_module, device)
    del model_prime
    return normgrad_map_np

# --- Metric Calculation Functions (definitions from Block 3) ---
def normalize_saliency_map_batch(saliency_maps_batch):
    normalized_maps = []
    for saliency_map in saliency_maps_batch:
        min_val, max_val = saliency_map.min(), saliency_map.max()
        if max_val - min_val > 1e-8: norm_map = (saliency_map - min_val) / (max_val - min_val)
        else: norm_map = torch.zeros_like(saliency_map)
        normalized_maps.append(norm_map)
    return torch.stack(normalized_maps)

def perturb_image_batch(images_batch, saliency_maps_batch, fraction_to_perturb, mode, baseline_value=0.0):
    B, C, H, W = images_batch.shape; perturbed_images_batch = images_batch.clone()
    saliency_maps_batch = saliency_maps_batch.to(images_batch.device, dtype=images_batch.dtype)
    saliency_maps_batch_norm = normalize_saliency_map_batch(saliency_maps_batch)
    saliency_flat = saliency_maps_batch_norm.view(B, -1)
    num_pixels_total = H * W; num_pixels_to_perturb = int(fraction_to_perturb * num_pixels_total)
    if num_pixels_to_perturb == 0 and fraction_to_perturb > 0: num_pixels_to_perturb = 1
    if num_pixels_to_perturb > num_pixels_total: num_pixels_to_perturb = num_pixels_total
    if num_pixels_to_perturb == 0 and mode == "delete": return perturbed_images_batch
    if num_pixels_to_perturb == num_pixels_total and mode == "insert": return perturbed_images_batch
    if num_pixels_to_perturb == 0 and mode == "insert": return torch.full_like(images_batch, baseline_value)
    if num_pixels_to_perturb == num_pixels_total and mode == "delete": return torch.full_like(images_batch, baseline_value)
    _, top_indices_flat = torch.topk(saliency_flat, num_pixels_to_perturb, dim=1, largest=True)
    perturb_mask_flat = torch.zeros_like(saliency_flat, dtype=torch.bool)
    perturb_mask_flat.scatter_(1, top_indices_flat, True)
    perturb_mask_2d = perturb_mask_flat.view(B, 1, H, W).expand_as(images_batch)
    if mode == 'delete': perturbed_images_batch = torch.where(perturb_mask_2d, torch.full_like(images_batch, baseline_value), images_batch)
    elif mode == 'insert':
        baseline_images = torch.full_like(images_batch, baseline_value)
        perturbed_images_batch = torch.where(perturb_mask_2d, images_batch, baseline_images)
    else: raise ValueError("mode must be 'delete' or 'insert'")
    return perturbed_images_batch

def get_batch_dice_score(model, images_batch, gt_masks_batch, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        images_batch = images_batch.to(device); gt_masks_batch = gt_masks_batch.to(device)
        outputs = model(images_batch); preds_prob = torch.sigmoid(outputs)
        preds_bin = (preds_prob > threshold).float()
        preds_flat = preds_bin.view(preds_bin.shape[0], -1)
        gt_flat = gt_masks_batch.view(gt_masks_batch.shape[0], -1)
        intersection = (preds_flat * gt_flat).sum(dim=1)
        dice_coeffs = (2. * intersection + 1e-6) / (preds_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1e-6)
        return dice_coeffs

def calculate_auc_metric_batch(model, images_orig_batch, gt_masks_batch, saliency_maps_batch,
                               device, mode, num_steps=20, baseline_value=0.0, threshold=0.5):
    fractions_perturbed = np.linspace(0, 1, num_steps + 1); batch_auc_scores = []
    for b_idx in range(images_orig_batch.shape[0]):
        image_orig = images_orig_batch[b_idx].unsqueeze(0)
        gt_mask = gt_masks_batch[b_idx].unsqueeze(0)
        saliency_map = saliency_maps_batch[b_idx].unsqueeze(0)
        scores_at_steps = []
        for fraction in fractions_perturbed:
            perturbed_image = perturb_image_batch(image_orig, saliency_map.squeeze(0), fraction, mode, baseline_value)
            dice_score_tensor = get_batch_dice_score(model, perturbed_image, gt_mask, device, threshold)
            scores_at_steps.append(dice_score_tensor.item())
        auc_value = sklearn_auc(fractions_perturbed, scores_at_steps)
        batch_auc_scores.append(auc_value)
    return torch.tensor(batch_auc_scores, device=device if torch.cuda.is_available() else "cpu")

def calculate_drop_in_performance_batch(model, images_orig_batch, gt_masks_batch, saliency_maps_batch,
                                        device, top_k_fraction=0.2, baseline_value=0.0, threshold=0.5):
    dice_orig_batch = get_batch_dice_score(model, images_orig_batch, gt_masks_batch, device, threshold)
    perturbed_images_batch = perturb_image_batch(images_orig_batch, saliency_maps_batch, top_k_fraction, 'delete', baseline_value)
    dice_perturbed_batch = get_batch_dice_score(model, perturbed_images_batch, gt_masks_batch, device, threshold)
    drop_batch = dice_orig_batch - dice_perturbed_batch
    return drop_batch

def calculate_pointing_game_accuracy_batch(saliency_maps_batch, gt_masks_batch):
    B, H, W = saliency_maps_batch.shape
    gt_masks_batch_squeezed = gt_masks_batch.squeeze(1) if gt_masks_batch.ndim == 4 else gt_masks_batch
    hits = 0
    for i in range(B):
        saliency_map_single = saliency_maps_batch[i]; gt_mask_single = gt_masks_batch_squeezed[i]
        max_val_idx_flat = torch.argmax(saliency_map_single.flatten())
        max_val_coords = np.unravel_index(max_val_idx_flat.cpu().numpy(), (H, W))
        if gt_mask_single[max_val_coords[0], max_val_coords[1]].item() > 0.5: # Check if it's 1
            hits += 1
    return torch.tensor([hits / B], device=saliency_maps_batch.device if saliency_maps_batch.is_cuda else "cpu")

# --- XAI Method Wrappers (definitions from Block 3) ---
def get_grad_cam_saliency(model, img_tensor_batch, device, target_layer_name, **kwargs):
    base_model_for_layers = model.model if hasattr(model,'model') else model
    target_layer_module = find_layer(base_model_for_layers, target_layer_name)
    if target_layer_module is None: raise ValueError(f"Grad-CAM: Target layer '{target_layer_name}' not found.")
    saliency_maps_list = []
    for i in range(img_tensor_batch.shape[0]):
        single_img_tensor = img_tensor_batch[i].unsqueeze(0)
        cam_map_np = gradcam_segmentation_data_intermediate(model, single_img_tensor, device, target_layer_module)
        if cam_map_np is None: cam_map_np = np.zeros(img_tensor_batch.shape[2:])
        saliency_maps_list.append(torch.from_numpy(cam_map_np).float())
    return torch.stack(saliency_maps_list).to(device)

def get_normgrad_saliency(model, img_tensor_batch, device, target_layer_name, **kwargs):
    base_model_for_layers = model.model if hasattr(model,'model') else model
    target_layer_module = find_layer(base_model_for_layers, target_layer_name)
    if target_layer_module is None: raise ValueError(f"NormGrad: Target layer '{target_layer_name}' not found.")
    saliency_maps_list = []; img_h, img_w = img_tensor_batch.shape[2:]
    for i in range(img_tensor_batch.shape[0]):
        single_img_tensor = img_tensor_batch[i].unsqueeze(0)
        ng_map_np = compute_normgrad_order0(model, single_img_tensor, target_layer_module, device)
        if ng_map_np is None: ng_map_np = np.zeros((img_h, img_w))
        else:
            if ng_map_np.shape != (img_h, img_w):
                ng_map_np = cv2.resize(ng_map_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        saliency_maps_list.append(torch.from_numpy(ng_map_np).float())
    return torch.stack(saliency_maps_list).to(device)

def get_normgrad_order1_saliency(model, img_tensor_batch, device, target_layer_name, adversarial=False, epsilon=0.0005, **kwargs):
    saliency_maps_list = []; img_h, img_w = img_tensor_batch.shape[2:]
    for i in range(img_tensor_batch.shape[0]):
        single_img_tensor = img_tensor_batch[i].unsqueeze(0)
        ng1_map_np = compute_normgrad_order1_by_name(model, single_img_tensor, target_layer_name, device, epsilon, adversarial)
        if ng1_map_np is None:
            logger.warning(f"NormGrad O1 failed for item {i}, layer {target_layer_name}. Using zeros.")
            ng1_map_np = np.zeros((img_h, img_w))
        else:
            if ng1_map_np.shape != (img_h, img_w):
                try: ng1_map_np = cv2.resize(ng1_map_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                except Exception as resize_e:
                    logger.warning(f"Could not resize NG1 map for item {i}. Error: {resize_e}. Using zeros.")
                    ng1_map_np = np.zeros((img_h, img_w))
        saliency_maps_list.append(torch.from_numpy(ng1_map_np).float())
    return torch.stack(saliency_maps_list).to(device)

# --- Combined Saliency Functions (definitions from Block 3) ---
def normalize_heatmap_tensor(heatmap_tensor): # Single H,W tensor
    min_val, max_val = heatmap_tensor.min(), heatmap_tensor.max()
    if max_val - min_val > 1e-8: return (heatmap_tensor - min_val) / (max_val - min_val)
    else: return torch.zeros_like(heatmap_tensor)

def get_combined_saliency(model, img_tensor_batch, device, target_layers_list, base_xai_method_fn, **kwargs):
    B, C, H, W = img_tensor_batch.shape
    combined_maps_batch = torch.zeros((B, H, W), device=device, dtype=torch.float32)
    for b_idx in range(B):
        single_img_tensor = img_tensor_batch[b_idx].unsqueeze(0)
        valid_maps_for_image = []; num_failed_layers = 0
        for layer_name in target_layers_list:
            try:
                saliency_map_single_layer_batch = base_xai_method_fn(model, single_img_tensor, device, layer_name, **kwargs)
                saliency_map_single_layer = saliency_map_single_layer_batch.squeeze(0)
                normalized_map = normalize_heatmap_tensor(saliency_map_single_layer)
                valid_maps_for_image.append(normalized_map)
            except Exception as e:
                logger.warning(f"Failed to get saliency for layer '{layer_name}', item {b_idx}. Error: {e}. Skipping.")
                num_failed_layers += 1
        if valid_maps_for_image:
            stacked_maps = torch.stack(valid_maps_for_image, dim=0)
            combined_map_single_image = torch.mean(stacked_maps, dim=0)
            combined_maps_batch[b_idx] = combined_map_single_image
        elif num_failed_layers == len(target_layers_list):
            logger.warning(f"All layers failed for item {b_idx}. Combined map is zeros.")
    return combined_maps_batch

# --- Main Evaluation Function for Faithfulness & Localization (definition from Block 3) ---
def evaluate_faithfulness_localization(
    model, dataloader, device, xai_method_fn,
    target_layers_or_name, num_batches_to_eval=None, auc_num_steps=10, drop_top_k_fraction=0.2,
    perturb_baseline_value=0.0, pred_threshold=0.5, **xai_kwargs):
    model.eval()
    target_desc = f"{len(target_layers_or_name)} layers (combined)" if isinstance(target_layers_or_name, list) else f"layer '{target_layers_or_name}'"
    results_dict = {"DAUC": [], "IAUC": [], "DropInPerformance": [], "PointingGameAccuracy": []}
    num_evaluated = 0
    
    # Ensure xai_method_fn has a __name__ attribute for logging
    xai_fn_name = getattr(xai_method_fn, '__name__', 'unknown_xai_method')
    if 'base_xai_method_fn' in xai_kwargs:
        base_xai_fn = xai_kwargs['base_xai_method_fn']
        base_xai_fn_name = getattr(base_xai_fn, '__name__', 'unknown_base_xai_method')
        desc_str = f"Eval XAI ({xai_fn_name} with {base_xai_fn_name} on {target_desc})"
    else:
        desc_str = f"Eval XAI ({xai_fn_name} on {target_desc})"

    max_desc_len = 70 # Limit tqdm output length
    if len(desc_str) > max_desc_len:
        desc_str = desc_str[:max_desc_len-3] + "..."

    # Use original sys.stdout for tqdm if it's a TTY, otherwise don't show progress bar if stdout is captured
    tqdm_file = sys.__stdout__ if sys.__stdout__.isatty() else None 
    eval_loop = tqdm(dataloader, desc=desc_str, leave=False, file=tqdm_file, disable=tqdm_file is None)
    
    for batch_idx, (images_batch, gt_masks_batch) in enumerate(eval_loop):
        if num_batches_to_eval is not None and batch_idx >= num_batches_to_eval: break
        images_batch = images_batch.to(device); gt_masks_batch = gt_masks_batch.to(device)
        try:
            saliency_maps_batch = xai_method_fn(model, images_batch, device, target_layers_or_name, **xai_kwargs)
            if saliency_maps_batch.shape[1:] != images_batch.shape[2:]: # Resize safeguard
                resized_saliency_maps = []
                for s_map in saliency_maps_batch:
                    if s_map.ndim == 2:
                        resized_map_np = cv2.resize(s_map.cpu().numpy(), images_batch.shape[2:][::-1], interpolation=cv2.INTER_LINEAR)
                        resized_saliency_maps.append(torch.from_numpy(resized_map_np).to(device))
                    else: resized_saliency_maps.append(s_map)
                saliency_maps_batch = torch.stack(resized_saliency_maps)
                if saliency_maps_batch.shape[1:] != images_batch.shape[2:]:
                    logger.error(f"Saliency map shape mismatch after resize. Skipping batch {batch_idx}.")
                    continue

            dauc_b = calculate_auc_metric_batch(model, images_batch, gt_masks_batch, saliency_maps_batch, device, 'delete', auc_num_steps, perturb_baseline_value, pred_threshold)
            results_dict["DAUC"].extend(dauc_b.cpu().tolist())
            iauc_b = calculate_auc_metric_batch(model, images_batch, gt_masks_batch, saliency_maps_batch, device, 'insert', auc_num_steps, perturb_baseline_value, pred_threshold)
            results_dict["IAUC"].extend(iauc_b.cpu().tolist())
            drop_b = calculate_drop_in_performance_batch(model, images_batch, gt_masks_batch, saliency_maps_batch, device, drop_top_k_fraction, perturb_baseline_value, pred_threshold)
            results_dict["DropInPerformance"].extend(drop_b.cpu().tolist())
            pg_acc_b = calculate_pointing_game_accuracy_batch(saliency_maps_batch, (gt_masks_batch > 0.5).float())
            results_dict["PointingGameAccuracy"].extend(pg_acc_b.cpu().tolist())
            num_evaluated += images_batch.shape[0]
        except Exception as e:
            logger.error(f"Error in batch {batch_idx} for XAI eval: {e}")
            logger.error(traceback.format_exc())
            continue
    final_metrics = {key: np.mean([v for v in values if not np.isnan(v)]) if values else np.nan for key, values in results_dict.items()}
    logger.info(f"--- XAI Metric Eval Summary ({xai_fn_name} on {num_evaluated} samples) ---")
    logger.info(f"Target: {target_desc}")
    for name, val in final_metrics.items():
        interp = ""
        if name == "DAUC": interp = "(Lower is better)"
        elif name == "IAUC": interp = "(Higher is better)"
        elif name == "DropInPerformance": interp = "(Higher is better)"
        elif name == "PointingGameAccuracy": interp = "(Higher is better)"
        logger.info(f"  Average {name}: {val:.4f} {interp}")
    return final_metrics

# --- K-Fold Evaluation Logic (from Block 3) ---
def run_kfold_evaluation():
    logger.info("Starting k-fold model evaluation.")
    all_results_list = []

    affected_loader = None
    affected_dataset = None
    actual_samples_used = 0
    try:
        if not os.path.exists(dev_csv_path): raise FileNotFoundError(f"Dev CSV not found: {dev_csv_path}")
        original_dev_df = pd.read_csv(dev_csv_path)
        if not os.path.isdir(affected_img_dir_path): raise FileNotFoundError(f"Affected dir not found: {affected_img_dir_path}")
        affected_files_on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(affected_img_dir_path, '*.jpg'))}
        if not affected_files_on_disk: raise ValueError("No JPGs in affected dir.")
        
        filename_col = 'image_name'
        if filename_col not in original_dev_df.columns:
            if 'Image Index' in original_dev_df.columns: filename_col = 'Image Index'
            elif 'file_name' in original_dev_df.columns: filename_col = 'file_name'
            else: raise ValueError(f"Filename column (e.g., 'image_name') not found in {dev_csv_path}")
        
        affected_mask = original_dev_df[filename_col].isin(affected_files_on_disk)
        filtered_df_all_affected = original_dev_df[affected_mask].copy()

        if filtered_df_all_affected.empty: raise ValueError("No CSV entries match files in affected dir.")
        num_available_affected = len(filtered_df_all_affected)

        final_filtered_df = None
        if NUM_AFFECTED_SAMPLES_TO_USE_KFOLD is None or num_available_affected <= NUM_AFFECTED_SAMPLES_TO_USE_KFOLD:
            final_filtered_df = filtered_df_all_affected
            actual_samples_used = num_available_affected
        else:
            np.random.seed(RANDOM_SEED_FOR_SAMPLING_KFOLD)
            selected_indices = np.random.choice(filtered_df_all_affected.index, size=NUM_AFFECTED_SAMPLES_TO_USE_KFOLD, replace=False)
            final_filtered_df = filtered_df_all_affected.loc[selected_indices]
            actual_samples_used = NUM_AFFECTED_SAMPLES_TO_USE_KFOLD
        
        if final_filtered_df is None or final_filtered_df.empty: raise ValueError("Failed to select final samples.")
        logger.info(f"Selected {actual_samples_used} 'affected' samples for k-fold evaluation.")

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv", encoding='utf-8') as temp_csv_file_obj:
            final_filtered_df.to_csv(temp_csv_file_obj.name, index=False)
            temp_csv_path = temp_csv_file_obj.name
        
        affected_dataset = ObjectCXRSataset(temp_csv_path, affected_img_dir_path, test_transform, (IMG_SIZE, IMG_SIZE))
        eval_batch_size_eff = BATCH_SIZE // 2 if BATCH_SIZE > 1 else 1
        affected_loader = DataLoader(affected_dataset, batch_size=eval_batch_size_eff, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        logger.info(f"Created DataLoader for {len(affected_dataset)} 'affected' samples with batch size {eval_batch_size_eff} for k-fold.")
        
        if os.path.exists(temp_csv_path): os.remove(temp_csv_path)

    except Exception as e:
        logger.error(f"Error creating DataLoader for k-fold affected samples: {e}")
        logger.error(traceback.format_exc())
        return

    if not affected_loader or not affected_dataset:
        logger.error("Failed to create DataLoader for k-fold. Aborting evaluation.")
        return

    model_paths = sorted(glob.glob(os.path.join(MODEL_BASE_DIR_KFOLD, '*.pth')))
    if not model_paths:
        logger.error(f"No model .pth files found in {MODEL_BASE_DIR_KFOLD}. Aborting k-fold.")
        return
    logger.info(f"Found {len(model_paths)} models for k-fold evaluation: {model_paths}")

    current_run_layers_to_combine_kfold = LIST_OF_LAYERS_TO_COMBINE_KFOLD # Default

    for model_idx, model_path in enumerate(model_paths):
        model_name = os.path.basename(model_path)
        logger.info(f"--- Processing K-Fold Model {model_idx+1}/{len(model_paths)}: {model_name} ---")
        
        current_model_base_metrics = {'model_name': model_name}
        
        load_start_time = time.time()
        try:
            model_instance = SegmentationModel().to(device)
            model_instance.load_state_dict(torch.load(model_path, map_location=device))
            model_instance.eval()
            load_end_time = time.time()
            current_model_base_metrics['model_load_time_sec'] = load_end_time - load_start_time
            logger.info(f"Model {model_name} loaded in {current_model_base_metrics['model_load_time_sec']:.2f} seconds.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.error(traceback.format_exc())
            failure_row = {**current_model_base_metrics, 'xai_method': 'N/A', 'eval_time_sec': 0,
                           'DAUC': np.nan, 'IAUC': np.nan, 'DropInPerformance': np.nan, 'PointingGameAccuracy': np.nan,
                           'error': f'ModelLoadError: {e}'}
            all_results_list.append(failure_row)
            continue

        if model_idx == 0: # Verify layers only for the first model
            logger.info("Verifying layer names for combined heatmaps (k-fold) using the first loaded model...")
            temp_model_check = model_instance.model if hasattr(model_instance, 'model') else model_instance
            valid_layers = []
            for layer_n_check in LIST_OF_LAYERS_TO_COMBINE_KFOLD:
                if find_layer(temp_model_check, layer_n_check) is not None:
                    valid_layers.append(layer_n_check)
                else:
                    logger.warning(f"Layer '{layer_n_check}' NOT FOUND in model. It will be excluded from combined heatmaps for k-fold.")
            if not valid_layers:
                logger.error("CRITICAL: No valid layers found for combined heatmaps (k-fold). Aborting XAI for k-fold.")
                failure_row = {**current_model_base_metrics, 'xai_method': 'Combined_XAI_Setup', 'eval_time_sec': 0,
                               'error': 'NoValidLayersForCombinedHeatmap_KFold'}
                all_results_list.append(failure_row)
                del model_instance
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                return # Stop all further k-fold processing
            logger.info(f"Using {len(valid_layers)} valid layers for combined heatmaps (k-fold).")
            current_run_layers_to_combine_kfold = valid_layers
            del temp_model_check

        xai_eval_configs = [
            {
                "name": "Combined_Grad_CAM_KFold",
                "xai_method_fn": get_combined_saliency,
                "kwargs_for_eval_func": {"base_xai_method_fn": get_grad_cam_saliency}
            },
            {
                "name": "Combined_NormGrad_Order_0_KFold",
                "xai_method_fn": get_combined_saliency,
                "kwargs_for_eval_func": {"base_xai_method_fn": get_normgrad_saliency}
            },
            {
                "name": "Combined_NormGrad_Order_1_Train_KFold",
                "xai_method_fn": get_combined_saliency,
                "kwargs_for_eval_func": {
                    "base_xai_method_fn": lambda m, i, dv, ln, **kw: get_normgrad_order1_saliency(
                        m, i, dv, ln, adversarial=False, epsilon=NORMGRAD_EPSILON_FOR_EVAL_KFOLD, **kw
                    )
                }
            }
        ]
        xai_eval_configs[2]["kwargs_for_eval_func"]["base_xai_method_fn"].__name__ = "get_normgrad_order1_saliency_train_base_kfold"

        for config in xai_eval_configs:
            xai_name = config["name"]
            logger.info(f"Evaluating XAI method: {xai_name} for model {model_name}")
            eval_start_time = time.time()
            metrics = None
            try:
                metrics = evaluate_faithfulness_localization(
                    model=model_instance,
                    dataloader=affected_loader,
                    device=device,
                    xai_method_fn=config["xai_method_fn"],
                    target_layers_or_name=current_run_layers_to_combine_kfold,
                    num_batches_to_eval=NUM_BATCHES_FOR_XAI_EVAL_KFOLD,
                    auc_num_steps=AUC_STEPS_KFOLD,
                    drop_top_k_fraction=DROP_TOP_K_KFOLD,
                    **config["kwargs_for_eval_func"]
                )
                eval_end_time = time.time()
                logger.info(f"Evaluation for {xai_name} on {model_name} finished in {eval_end_time - eval_start_time:.2f} seconds.")
                
                row = {**current_model_base_metrics, 'xai_method': xai_name, 'eval_time_sec': eval_end_time - eval_start_time}
                if metrics: row.update(metrics)
                else: 
                    row.update({'DAUC': np.nan, 'IAUC': np.nan, 'DropInPerformance': np.nan, 'PointingGameAccuracy': np.nan, 'error': 'MetricsCalculationReturnedNone'})
                all_results_list.append(row)

            except Exception as e_xai:
                eval_end_time = time.time()
                logger.error(f"Error during XAI evaluation for {xai_name} on model {model_name}: {e_xai}")
                logger.error(traceback.format_exc())
                failure_row = {**current_model_base_metrics, 'xai_method': xai_name, 'eval_time_sec': eval_end_time - eval_start_time,
                               'DAUC': np.nan, 'IAUC': np.nan, 'DropInPerformance': np.nan, 'PointingGameAccuracy': np.nan,
                               'error': f'XAIMethodError: {e_xai}'}
                all_results_list.append(failure_row)
        
        del model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache.")

    if all_results_list:
        results_df = pd.DataFrame(all_results_list)
        try:
            results_df.to_csv(RESULTS_CSV_PATH, index=False)
            logger.info(f"All k-fold evaluation results saved to: {RESULTS_CSV_PATH}")
        except Exception as e_csv:
            logger.error(f"Failed to save k-fold results to CSV {RESULTS_CSV_PATH}: {e_csv}")
            logger.error("Dumping k-fold results to log instead:\n" + results_df.to_string())
    else:
        logger.warning("No k-fold results were generated to save to CSV.")

    logger.info("--- K-Fold Model Evaluation Script Finished ---")


# --- Example Usage V4 (from Block 2 of the provided snippet) ---
# This example demonstrates using the evaluate_faithfulness_localization 
# with combined heatmaps, using a specific setup from the original snippet.
# To run this, you would need to ensure 'model', 'device', 'ObjectCXRSataset',
# 'test_transform', 'IMG_SIZE', 'BATCH_SIZE', 'NORMGRAD_EPSILON' are appropriately 
# defined in the global scope if you call this function directly, 
# or modify this function to accept them as parameters.

def run_faithfulness_localization_example_v4(model_to_eval, device_to_use, base_data_path_for_example, affected_img_subdir_for_example, dev_csv_filename_for_example, dev_img_dir_name_for_example, img_size_for_example, batch_size_for_example, normgrad_epsilon_for_example):
    logger.info("\n--- Starting Faithfulness & Localization Metrics Cell (Example V4) ---")
    
    # --- Configuration (specific to this example, from Block 2) ---
    LIST_OF_LAYERS_TO_COMBINE_V4 = [
        'encoder.conv1',
        'encoder.maxpool',
        'encoder.layer1',
        # ... (rest of the layers from Block 2's LIST_OF_LAYERS_TO_COMBINE)
        # For brevity, using a subset. The full list is very long.
        # Ensure these layers are valid for the model_to_eval being used.
        'encoder.layer2',
        'encoder.layer3',
        'encoder.layer4',
        'decoder.aspp',
        'segmentation_head.2.activation'
    ]
    # IMPORTANT: Verify these layer names exist in your specific model!
    logger.info(f"Example V4: Verifying layers to combine for the provided model...")
    temp_model_check_v4 = model_to_eval.model if hasattr(model_to_eval, 'model') else model_to_eval
    verified_layers_v4 = []
    for layer_n in LIST_OF_LAYERS_TO_COMBINE_V4:
        if find_layer(temp_model_check_v4, layer_n) is not None:
            verified_layers_v4.append(layer_n)
            # logger.info(f" - {layer_n}: Found")
        else:
            logger.warning(f" - {layer_n}: NOT FOUND! Will be excluded from Example V4.")
    del temp_model_check_v4
    if not verified_layers_v4:
        logger.error("ERROR: No valid layers in LIST_OF_LAYERS_TO_COMBINE_V4 were found for Example V4. Aborting example.")
        return
    LIST_OF_LAYERS_TO_COMBINE_V4 = verified_layers_v4
    logger.info(f"Example V4: Using {len(LIST_OF_LAYERS_TO_COMBINE_V4)} verified layers.")

    NUM_AFFECTED_SAMPLES_TO_USE_V4 = 500 
    RANDOM_SEED_FOR_SAMPLING_V4 = 42

    affected_img_dir_path_v4 = os.path.join(base_data_path_for_example, dev_img_dir_name_for_example, affected_img_subdir_for_example)
    dev_csv_path_v4 = os.path.join(base_data_path_for_example, dev_csv_filename_for_example)

    logger.info(f"Example V4: Combining heatmaps from {len(LIST_OF_LAYERS_TO_COMBINE_V4)} layers.")
    logger.info(f"Example V4: Targeting images in: {affected_img_dir_path_v4}")
    if NUM_AFFECTED_SAMPLES_TO_USE_V4 is not None:
        logger.info(f"Example V4: Attempting to use a subset of {NUM_AFFECTED_SAMPLES_TO_USE_V4} 'affected' samples.")
    else:
        logger.info("Example V4: Attempting to use all available 'affected' samples.")

    # --- Create DataLoader for the 'affected' subset (logic from Block 2) ---
    affected_loader_v4 = None
    affected_dataset_v4 = None
    temp_csv_file_v4 = None
    actual_samples_used_v4 = 0

    try:
        if not os.path.exists(dev_csv_path_v4):
             raise FileNotFoundError(f"Example V4: Dev CSV not found at: {dev_csv_path_v4}")
        original_dev_df_v4 = pd.read_csv(dev_csv_path_v4)
        
        if not os.path.isdir(affected_img_dir_path_v4):
             raise FileNotFoundError(f"Example V4: The 'affected' directory not found at: {affected_img_dir_path_v4}")
        affected_files_on_disk_paths_v4 = glob.glob(os.path.join(affected_img_dir_path_v4, '*.jpg')) # Assuming JPG
        affected_filenames_on_disk_v4 = {os.path.basename(p) for p in affected_files_on_disk_paths_v4}
        if not affected_filenames_on_disk_v4: raise ValueError(f"Example V4: No JPG files found on disk in {affected_img_dir_path_v4}.")

        filename_column_v4 = 'image_name' # Adjust if needed for your CSV structure
        if filename_column_v4 not in original_dev_df_v4.columns: raise ValueError(f"Example V4: Column '{filename_column_v4}' not found in {dev_csv_path_v4}")
        
        affected_mask_v4 = original_dev_df_v4[filename_column_v4].isin(affected_filenames_on_disk_v4)
        filtered_df_all_affected_v4 = original_dev_df_v4[affected_mask_v4].copy()
        if filtered_df_all_affected_v4.empty: raise ValueError(f"Example V4: No filenames in CSV matched files in the 'affected' directory.")
        num_available_affected_v4 = len(filtered_df_all_affected_v4)

        final_filtered_df_v4 = None
        if NUM_AFFECTED_SAMPLES_TO_USE_V4 is None or num_available_affected_v4 <= NUM_AFFECTED_SAMPLES_TO_USE_V4:
            final_filtered_df_v4 = filtered_df_all_affected_v4
            actual_samples_used_v4 = num_available_affected_v4
        else:
            np.random.seed(RANDOM_SEED_FOR_SAMPLING_V4)
            selected_indices_v4 = np.random.choice(filtered_df_all_affected_v4.index, size=NUM_AFFECTED_SAMPLES_TO_USE_V4, replace=False)
            final_filtered_df_v4 = filtered_df_all_affected_v4.loc[selected_indices_v4]
            actual_samples_used_v4 = NUM_AFFECTED_SAMPLES_TO_USE_V4
        if final_filtered_df_v4 is None or final_filtered_df_v4.empty: raise ValueError("Example V4: Failed to select final samples for the dataset.")
        logger.info(f"Example V4: Selected {actual_samples_used_v4} samples for evaluation.")

        temp_csv_file_obj_v4 = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv", encoding='utf-8')
        final_filtered_df_v4.to_csv(temp_csv_file_obj_v4.name, index=False)
        temp_csv_path_v4 = temp_csv_file_obj_v4.name
        temp_csv_file_obj_v4.close()

        # Use the globally defined ObjectCXRSataset and test_transform
        current_test_transform_v4 = A.Compose([A.Resize(img_size_for_example, img_size_for_example)])
        affected_dataset_v4 = ObjectCXRSataset(
            csv_file=temp_csv_path_v4,
            img_dir=affected_img_dir_path_v4, 
            transform=current_test_transform_v4,
            img_size=(img_size_for_example, img_size_for_example)
        )

        eval_batch_size_v4 = batch_size_for_example // 2 if batch_size_for_example > 1 else 1
        affected_loader_v4 = DataLoader(affected_dataset_v4, batch_size=eval_batch_size_v4, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        logger.info(f"Example V4: Created DataLoader for the selected {len(affected_dataset_v4)} 'affected' samples.")

    except FileNotFoundError as fnf_err: logger.error(f"Example V4: Error creating affected_loader: {fnf_err}"); affected_loader_v4 = None
    except ValueError as val_err: logger.error(f"Example V4: Error creating affected_loader: {val_err}"); affected_loader_v4 = None
    except Exception as e: logger.error(f"Example V4: Unexpected error creating affected_loader: {e}"); traceback.print_exc(); affected_loader_v4 = None
    finally:
        if 'temp_csv_path_v4' in locals() and temp_csv_path_v4 and os.path.exists(temp_csv_path_v4):
            try: os.remove(temp_csv_path_v4)
            except Exception as e: logger.warning(f"Example V4: Could not remove temporary CSV file {temp_csv_path_v4}: {e}")

    if affected_loader_v4 and affected_dataset_v4:
        logger.info(f"\n--- Example V4: Starting Faithfulness & Localization Metrics Evaluation using COMBINED Heatmaps ---")
        logger.info(f"--- Example V4: Target Layers ({len(LIST_OF_LAYERS_TO_COMBINE_V4)}): {LIST_OF_LAYERS_TO_COMBINE_V4[:3]}... ---")
        logger.info(f"--- Example V4: Evaluating on {actual_samples_used_v4} selected 'affected' Samples ---")

        NUM_BATCHES_FOR_XAI_EVAL_V4 = None 
        AUC_STEPS_V4 = 10
        DROP_TOP_K_V4 = 0.2
        NORMGRAD_EPSILON_FOR_EVAL_V4 = normgrad_epsilon_for_example

        combined_grad_cam_metrics_v4 = None
        combined_normgrad0_metrics_v4 = None
        combined_normgrad1_train_metrics_v4 = None

        logger.info("\nExample V4: Evaluating with Combined Grad-CAM...")
        try:
            combined_grad_cam_metrics_v4 = evaluate_faithfulness_localization(
                model=model_to_eval,
                dataloader=affected_loader_v4,
                device=device_to_use,
                xai_method_fn=get_combined_saliency,
                target_layers_or_name=LIST_OF_LAYERS_TO_COMBINE_V4,
                num_batches_to_eval=NUM_BATCHES_FOR_XAI_EVAL_V4,
                auc_num_steps=AUC_STEPS_V4,
                drop_top_k_fraction=DROP_TOP_K_V4,
                base_xai_method_fn=get_grad_cam_saliency
            )
        except Exception as e:
            logger.error(f"Example V4: Error during Combined Grad-CAM metric evaluation: {e}")
            traceback.print_exc()

        logger.info("\nExample V4: Evaluating with Combined NormGrad Order 0...")
        try:
            combined_normgrad0_metrics_v4 = evaluate_faithfulness_localization(
                model=model_to_eval,
                dataloader=affected_loader_v4,
                device=device_to_use,
                xai_method_fn=get_combined_saliency,
                target_layers_or_name=LIST_OF_LAYERS_TO_COMBINE_V4,
                num_batches_to_eval=NUM_BATCHES_FOR_XAI_EVAL_V4,
                auc_num_steps=AUC_STEPS_V4,
                drop_top_k_fraction=DROP_TOP_K_V4,
                base_xai_method_fn=get_normgrad_saliency
            )
        except Exception as e:
            logger.error(f"Example V4: Error during Combined NormGrad Order 0 metric evaluation: {e}")
            traceback.print_exc()

        logger.info("\nExample V4: Evaluating with Combined NormGrad Order 1 (Training)...")
        try:
            get_ng1_train_base_fn_v4 = lambda m, img, d, layer, **kwargs: get_normgrad_order1_saliency(
                m, img, d, layer, adversarial=False, epsilon=NORMGRAD_EPSILON_FOR_EVAL_V4, **kwargs
            )
            get_ng1_train_base_fn_v4.__name__ = "get_normgrad_order1_saliency_train_base_v4"

            combined_normgrad1_train_metrics_v4 = evaluate_faithfulness_localization(
                model=model_to_eval,
                dataloader=affected_loader_v4,
                device=device_to_use,
                xai_method_fn=get_combined_saliency,
                target_layers_or_name=LIST_OF_LAYERS_TO_COMBINE_V4,
                num_batches_to_eval=NUM_BATCHES_FOR_XAI_EVAL_V4,
                auc_num_steps=AUC_STEPS_V4,
                drop_top_k_fraction=DROP_TOP_K_V4,
                base_xai_method_fn=get_ng1_train_base_fn_v4
            )
        except Exception as e:
            logger.error(f"Example V4: Error during Combined NormGrad Order 1 (Train) metric evaluation: {e}")
            traceback.print_exc()

        logger.info(f"\n\n--- Example V4: XAI Metric Comparison (COMBINED Heatmaps from {len(LIST_OF_LAYERS_TO_COMBINE_V4)} layers) ---")
        logger.info(f"--- Example V4: Evaluated on {actual_samples_used_v4} selected 'affected' samples ---")
        results_found_v4 = False
        if combined_grad_cam_metrics_v4:
            logger.info("\nExample V4: Combined Grad-CAM Metrics:")
            for k, v in combined_grad_cam_metrics_v4.items(): logger.info(f"  {k}: {v:.4f}")
            results_found_v4 = True
        if combined_normgrad0_metrics_v4:
            logger.info("\nExample V4: Combined NormGrad Order 0 Metrics:")
            for k, v in combined_normgrad0_metrics_v4.items(): logger.info(f"  {k}: {v:.4f}")
            results_found_v4 = True
        if combined_normgrad1_train_metrics_v4:
            logger.info("\nExample V4: Combined NormGrad Order 1 (Train) Metrics:")
            for k, v in combined_normgrad1_train_metrics_v4.items(): logger.info(f"  {k}: {v:.4f}")
            results_found_v4 = True

        if not results_found_v4:
            logger.info("Example V4: No combined XAI metrics were successfully computed.")
    else:
        logger.info("\nExample V4: Skipping Faithfulness/Localization evaluation using combined heatmaps because the DataLoader could not be created.")

    logger.info("\n--- Faithfulness and Localization Metrics Cell (Example V4) Finished ---")


# --- Script Entry Point (for K-Fold Evaluation by default) ---
if __name__ == "__main__":
    essential_globals = ['IMG_SIZE', 'BATCH_SIZE', 'NORMGRAD_EPSILON', 'BASE_DATA_PATH', 'device', 'test_transform']
    missing_globals = [g for g in essential_globals if g not in globals() and g not in locals()]
    
    # Check if essential globals are defined (they are defined at the script level now)
    # So this check might always pass if the script is run directly.
    # The original check was for a notebook context.
    
    # For run_kfold_evaluation, globals like IMG_SIZE, BATCH_SIZE etc. are defined at the top of the script.
    # MODEL_BASE_DIR_KFOLD also needs to be a valid path.
    if not os.path.isdir(MODEL_BASE_DIR_KFOLD):
        logger.error(f"MODEL_BASE_DIR_KFOLD '{MODEL_BASE_DIR_KFOLD}' is not a valid directory. K-fold evaluation cannot run.")
    else:
        run_kfold_evaluation()

    # To run the Example V4, you would typically call it after setting up the required parameters:
    # logger.info("Preparing to run Example V4...")
    # # Example setup for run_faithfulness_localization_example_v4:
    # # 1. Define device (already global)
    # # 2. Define IMG_SIZE, BATCH_SIZE, NORMGRAD_EPSILON (already global, or override)
    # example_img_size = 256
    # example_batch_size = 8 
    # example_normgrad_epsilon = 0.0005
    # # 3. Define paths for the example data
    # example_base_data_path = "/home/linati/object-CXR_EB0/object-CXR" # Adjust as needed
    # example_affected_subdir = "affected"
    # example_dev_csv = "dev.csv"
    # example_dev_img_dir = "dev"
    # # 4. Load your model for the example
    # # example_model_path = "path_to_your_specific_model_for_example.pth" 
    # # if os.path.exists(example_model_path):
    # #     logger.info(f"Loading model for Example V4 from: {example_model_path}")
    # #     example_model = SegmentationModel().to(device)
    # #     example_model.load_state_dict(torch.load(example_model_path, map_location=device))
    # #     example_model.eval()
    # #     run_faithfulness_localization_example_v4(
    # #         model_to_eval=example_model, 
    # #         device_to_use=device, 
    # #         base_data_path_for_example=example_base_data_path,
    # #         affected_img_subdir_for_example=example_affected_subdir,
    # #         dev_csv_filename_for_example=example_dev_csv,
    # #         dev_img_dir_name_for_example=example_dev_img_dir,
    # #         img_size_for_example=example_img_size,
    # #         batch_size_for_example=example_batch_size,
    # #         normgrad_epsilon_for_example=example_normgrad_epsilon
    # #     )
    # #     del example_model
    # #     if torch.cuda.is_available(): torch.cuda.empty_cache()
    # # else:
    # #     logger.error(f"Model path for Example V4 not found: {example_model_path}. Cannot run Example V4.")
    # logger.info("Example V4 run (commented out by default) finished consideration.")

logger.info("--- XAI Evaluation Framework Script Finished ---")
