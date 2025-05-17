import os
import shutil
import random
import yaml
import time
import traceback
from typing import Tuple, List, Optional, Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms

from torchmetrics import AUROC, Accuracy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import peak_local_max

# -------------------- Configuration --------------------
NORMAL_FOLDER_NAME = 'normal'
AFFECTED_FOLDER_NAME = 'affected'

DRIVE_BASE = '/home/linati/object-CXR_EB0'
DATASET_BASE_PATH = '/home/linati/object-CXR_EB0/object-CXR'
ANNOTATIONS_CSV_PATH = "/home/linati/object-CXR_EB0/object-CXR/dev.csv"

class Config:
    def __init__(self, drive_base: str = DRIVE_BASE, dataset_base: str = DATASET_BASE_PATH):
        self.model_config = {
            'name': 'resnet34',
            'num_classes': 2,
            'pretrained': True,
            'input_size': 600
        }
        self.training_config = {
            'batch_size': 16,
            'epochs': 20,
            'lr': 0.005,
            'momentum': 0.9,
            'scheduler_step': 5,
            'scheduler_gamma': 0.1
        }
        base_checkpoint_dir = os.path.join(drive_base, 'XIQA_ObjectCXR_EfficientNetB0_Checkpoints')
        base_results_dir = os.path.join(drive_base, 'results_object_cxr_resnet34')
        if dataset_base is None or not os.path.isdir(dataset_base):
            dataset_base = './object-CXR_fallback'
            os.makedirs(os.path.join(dataset_base, 'train'), exist_ok=True)
            os.makedirs(os.path.join(dataset_base, 'dev'), exist_ok=True)
            if not os.path.exists(os.path.join(dataset_base, 'train.csv')):
                pd.DataFrame(columns=['image_name', 'annotation']).to_csv(os.path.join(dataset_base, 'train.csv'), index=False)
            if not os.path.exists(os.path.join(dataset_base, 'dev.csv')):
                pd.DataFrame(columns=['image_name', 'annotation']).to_csv(os.path.join(dataset_base, 'dev.csv'), index=False)
        self.paths = {
            'checkpoints': base_checkpoint_dir,
            'results': base_results_dir,
            'data_root': os.path.join(dataset_base, 'train'),
            'test_data_root': os.path.join(dataset_base, 'dev'),
            'train_csv': os.path.join(dataset_base, 'train.csv'),
            'dev_csv': os.path.join(dataset_base, 'dev.csv'),
            'train_normal': os.path.join(dataset_base, 'train', NORMAL_FOLDER_NAME),
            'train_affected': os.path.join(dataset_base, 'train', AFFECTED_FOLDER_NAME),
            'test_normal': os.path.join(dataset_base, 'dev', NORMAL_FOLDER_NAME),
            'test_affected': os.path.join(dataset_base, 'dev', AFFECTED_FOLDER_NAME),
        }
        for key, path in self.paths.items():
            if key in ['checkpoints', 'results']:
                os.makedirs(path, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Helper Classes --------------------
class DictAsMember(dict):
    def __getattr__(self, name):
        try: return self[name]
        except KeyError: raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name):
        try: del self[name]
        except KeyError: raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# -------------------- Model and XAI Classes --------------------
def get_transforms(is_training: bool, input_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize
        ])

def get_bounding_boxes_from_annotation(ann_str: str, original_img_width: int, original_img_height: int) -> List[Tuple[int, int, int, int]]:
    bounding_boxes = []
    if not isinstance(ann_str, str) or ann_str.strip() == "" or ann_str.lower() == 'nan':
        return bounding_boxes
    try:
        objects = ann_str.split(';')
        for obj in objects:
            if not obj.strip(): continue
            try:
                parts = list(map(int, obj.split()))
                obj_type = parts[0]
                coords = parts[1:]
            except ValueError:
                continue
            x1, y1, x2, y2 = -1, -1, -1, -1
            if obj_type == 0 and len(coords) == 4:
                x1_obj, y1_obj, x2_obj, y2_obj = coords
                x1 = min(x1_obj, x2_obj)
                y1 = min(y1_obj, y2_obj)
                x2 = max(x1_obj, x2_obj)
                y2 = max(y1_obj, y2_obj)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, original_img_width)
                y2 = min(y2, original_img_height)
            elif obj_type == 1 and len(coords) == 4:
                x1_obj, y1_obj, x2_obj, y2_obj = coords
                x1 = min(x1_obj, x2_obj)
                y1 = min(y1_obj, y2_obj)
                x2 = max(x1_obj, x2_obj)
                y2 = max(y1_obj, y2_obj)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, original_img_width)
                y2 = min(y2, original_img_height)
            elif obj_type == 2 and len(coords) >= 6 and len(coords) % 2 == 0:
                points = np.array(coords).reshape(-1, 2)
                points[:, 0] = np.clip(points[:, 0], 0, original_img_width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, original_img_height - 1)
                x1_poly, y1_poly = points.min(axis=0)
                x2_poly, y2_poly = points.max(axis=0)
                x1, y1, x2, y2 = int(x1_poly), int(y1_poly), int(x2_poly), int(y2_poly)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, original_img_width)
                y2 = min(y2, original_img_height)
            if x1 < x2 and y1 < y2:
                bounding_boxes.append((x1, y1, x2, y2))
    except Exception:
        traceback.print_exc()
    return bounding_boxes

class ImageQualityModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.model_name = config.model_config.get('name', 'resnet34')
        num_classes = config.model_config.get('num_classes', 2)
        pretrained = config.model_config.get('pretrained', True)
        if self.model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet34(weights=weights)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
    def forward(self, x):
        return self.model(x)

class GradCam:
    def __init__(self, model: nn.Module, model_args: DictAsMember, device: torch.device, input_size: Tuple[int, int]):
        self.model = model.eval()
        self.device = device
        self.input_size = input_size
        self.target_layer = None
        self.gradients = None
        self.activations = None
        self.hooks = []
        model_name = model_args.get('name', 'unknown')
        target_layer_name = model_args.get('target_layer', None)
        if target_layer_name is None:
            if model_name == 'resnet34': target_layer_name = 'model.layer4'
            else: return
        try:
            current_obj = self.model
            for part in target_layer_name.split('.'):
                current_obj = getattr(current_obj, part)
            self.target_layer = current_obj
            handle_fw = self.target_layer.register_forward_hook(self._forward_hook)
            handle_bw = self.target_layer.register_full_backward_hook(self._backward_hook)
            self.hooks.extend([handle_fw, handle_bw])
        except Exception:
            traceback.print_exc()
    def _forward_hook(self, module, input, output):
        act = output[0] if isinstance(output, tuple) and isinstance(output[0], torch.Tensor) else output
        if isinstance(act, torch.Tensor): self.activations = act.detach()
    def _backward_hook(self, module, grad_input, grad_output):
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach()
    def _normalize_cam(self, cam_tensor):
        if cam_tensor is None or cam_tensor.numel() == 0: return torch.zeros_like(cam_tensor) if cam_tensor is not None else None
        norm_cams = []
        for i in range(cam_tensor.shape[0]):
            cam_single = cam_tensor[i]
            min_val, max_val = torch.min(cam_single), torch.max(cam_single)
            range_val = max_val - min_val
            norm_cam = (cam_single - min_val) / (range_val + 1e-8) if range_val > 1e-7 else torch.zeros_like(cam_single)
            norm_cams.append(norm_cam)
        return torch.stack(norm_cams)
    def generate_cam(self, input_tensor, target_class=None):
        if not self.hooks or self.target_layer is None: return None, None
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3: input_tensor = input_tensor.unsqueeze(0)
        batch_size = input_tensor.shape[0]
        self.gradients, self.activations = None, None
        try:
            input_tensor.requires_grad_(True)
            model_output = self.model(input_tensor)
        except Exception:
            traceback.print_exc(); return None, None
        target_classes = torch.argmax(model_output, dim=1) if target_class is None else torch.tensor([target_class] * batch_size, device=self.device)
        try:
            self.model.zero_grad()
            one_hot_output = F.one_hot(target_classes, num_classes=model_output.shape[-1]).float()
            model_output.backward(gradient=one_hot_output, retain_graph=False)
        except Exception:
            traceback.print_exc(); return None, None
        if self.activations is None or self.gradients is None: return None, None
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam_4d = cam.unsqueeze(1)
        h_out, w_out = self.input_size
        try:
            heatmap_resized = F.interpolate(cam_4d, size=(h_out, w_out), mode='bilinear', align_corners=False)
        except Exception:
            return None, None
        heatmap_final = self._normalize_cam(heatmap_resized.squeeze(1))
        heatmap_np = heatmap_final.cpu().numpy()
        target_classes_np = target_classes.cpu().numpy()
        return heatmap_np, target_classes_np
    def remove_hooks(self):
        for handle in self.hooks: handle.remove()
        self.hooks = []
    def __del__(self): self.remove_hooks()

class NormGrad:
    def __init__(self, model: nn.Module, model_args: DictAsMember, input_size: Tuple[int, int]):
        self.model = model.eval()
        self.input_size = input_size
        self.device = next(model.parameters()).device
        self.hooks = []
        self.target_layers_names = []
        self.gradients: Dict[str, Optional[torch.Tensor]] = {}
        self.activations: Dict[str, Optional[torch.Tensor]] = {}
        model_name = model_args.get('name', 'unknown')
        target_layers_config = model_args.get('target_layers', None)
        if isinstance(target_layers_config, list) and all(isinstance(name, str) for name in target_layers_config):
            self.target_layers_names = target_layers_config
        elif isinstance(target_layers_config, str): self.target_layers_names = [target_layers_config]
        else:
            if model_name == 'resnet34': self.target_layers_names = ['model.layer4']
            else: return
        for layer_name in self.target_layers_names:
            try:
                current_obj = self.model
                for part in layer_name.split('.'):
                    current_obj = getattr(current_obj, part)
                target_layer = current_obj
                handle_fw = target_layer.register_forward_hook(
                    lambda module, input, output, name=layer_name: self._forward_hook(module, input, output, name)
                )
                handle_bw = target_layer.register_full_backward_hook(
                    lambda module, grad_in, grad_out, name=layer_name: self._backward_hook(module, grad_in, grad_out, name)
                )
                self.hooks.extend([handle_fw, handle_bw])
            except Exception:
                traceback.print_exc()
    def _forward_hook(self, module, input, output, name):
        act = output[0] if isinstance(output, tuple) and isinstance(output[0], torch.Tensor) else output
        if isinstance(act, torch.Tensor): self.activations[name] = act.detach()
    def _backward_hook(self, module, grad_in, grad_out, name):
        if grad_out and grad_out[0] is not None:
            self.gradients[name] = grad_out[0].detach()
    def _normalize_cam(self, cam_tensor):
        if cam_tensor is None or cam_tensor.numel() == 0: return torch.zeros_like(cam_tensor) if cam_tensor is not None else None
        norm_cams = []
        for i in range(cam_tensor.shape[0]):
            cam_single = cam_tensor[i]
            min_val, max_val = torch.min(cam_single), torch.max(cam_single)
            range_val = max_val - min_val
            norm_cam = (cam_single - min_val) / (range_val + 1e-8) if range_val > 1e-7 else torch.zeros_like(cam_single)
            norm_cams.append(norm_cam)
        return torch.stack(norm_cams)
    def __call__(self, input_tensor, method='scaling', target_class=None):
        if not self.hooks or not self.target_layers_names: return None, None
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        if input_tensor.dim() == 3: input_tensor = input_tensor.unsqueeze(0)
        batch_size = input_tensor.shape[0]
        for layer_name in self.target_layers_names:
            self.gradients[layer_name] = None
            self.activations[layer_name] = None
        try:
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
        except Exception:
            traceback.print_exc(); return None, None
        target_classes = torch.argmax(output, dim=1) if target_class is None else torch.tensor([target_class] * batch_size, device=self.device)
        try:
            self.model.zero_grad()
            one_hot = F.one_hot(target_classes, num_classes=output.shape[-1]).float()
            output.backward(gradient=one_hot, retain_graph=False)
        except Exception:
            traceback.print_exc(); return None, None
        final_cam_batch_np = None
        h_out, w_out = self.input_size
        for layer_name in self.target_layers_names:
            target = self.activations.get(layer_name)
            grad_init = self.gradients.get(layer_name)
            if target is None or grad_init is None: continue
            if target.dim() != 4 or grad_init.dim() != 4: continue
            out = None
            if method == 'scaling':
                out = -target * grad_init
            elif method == 'conv1x1':
                target_norm = torch.norm(target, p=2, dim=1, keepdim=True)
                grad_norm = torch.norm(grad_init, p=2, dim=1, keepdim=True)
                out = -target_norm * grad_norm
            elif method == 'conv3x3':
                try:
                    N, C_t, H_t, W_t = target.shape
                    unfold_act_flat = F.unfold(target, kernel_size=3, padding=1)
                    unfold_act_reshaped = unfold_act_flat.view(N, C_t * 9, H_t, W_t)
                    unfold_act_norm = torch.norm(unfold_act_reshaped, p=2, dim=1, keepdim=True)
                    grad_norm = torch.norm(grad_init, p=2, dim=1, keepdim=True)
                    out = -unfold_act_norm * grad_norm
                except Exception:
                    traceback.print_exc()
                    out = -target * grad_init
            else:
                out = -target * grad_init
            if out is None: continue
            cam_layer = torch.norm(out, p=2, dim=1, keepdim=True)
            try:
                cam_resized = F.interpolate(cam_layer, size=(h_out, w_out), mode='bilinear', align_corners=False)
            except Exception:
                continue
            cam_normalized_torch = self._normalize_cam(cam_resized.squeeze(1))
            current_layer_cam_np = cam_normalized_torch.cpu().numpy()
            if final_cam_batch_np is None:
                final_cam_batch_np = current_layer_cam_np.copy()
            else:
                for i in range(current_layer_cam_np.shape[0]):
                    if not np.allclose(current_layer_cam_np[i], 0, atol=1e-5):
                        final_cam_batch_np[i] *= current_layer_cam_np[i]
        if final_cam_batch_np is None: return None, None
        final_cam_normalized_np = np.zeros_like(final_cam_batch_np)
        for i in range(final_cam_batch_np.shape[0]):
            img_cam = final_cam_batch_np[i]
            min_val, max_val = np.min(img_cam), np.max(img_cam)
            range_val = max_val - min_val
            if range_val > 1e-8:
                final_cam_normalized_np[i] = (img_cam - min_val) / range_val
        target_classes_np = target_classes.cpu().numpy()
        return final_cam_normalized_np, target_classes_np
    def remove_hooks(self):
        for handle in self.hooks: handle.remove()
        self.hooks = []
    def __del__(self): self.remove_hooks()

# -------------------- Pointing Game Logic --------------------
def pointing_game_accuracy(heatmap, bboxes, orig_w, orig_h, tolerance=0):
    if not bboxes: return 0.0
    if heatmap is None or heatmap.size == 0: return 0.0
    h_min, h_max = np.min(heatmap), np.max(heatmap)
    if h_max - h_min > 1e-6:
        norm_heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        norm_heatmap = np.zeros_like(heatmap)
    y_max, x_max = np.unravel_index(np.argmax(norm_heatmap), norm_heatmap.shape)
    heat_h, heat_w = norm_heatmap.shape
    if orig_w == 0 or orig_h == 0: return 0.0
    scale_x = heat_w / orig_w
    scale_y = heat_h / orig_h
    for x1_orig, y1_orig, x2_orig, y2_orig in bboxes:
        x1_s = x1_orig * scale_x - tolerance
        y1_s = y1_orig * scale_y - tolerance
        x2_s = x2_orig * scale_x + tolerance
        y2_s = y2_orig * scale_y + tolerance
        if (x1_s <= x_max <= x2_s) and (y1_s <= y_max <= y2_s):
            return 1.0
    return 0.0

# -------------------- Visualization and Evaluation --------------------
def visualize_and_pg(image_path, visualizer, save_path=None):
    resized_image_for_display, input_tensor, orig_dims = visualizer.preprocess_image(image_path)
    if resized_image_for_display is None or input_tensor is None or orig_dims is None:
        print(f"Could not preprocess image: {image_path}. Skipping visualization.")
        return None
    original_width, original_height = orig_dims
    input_tensor_batch = input_tensor.unsqueeze(0).to(visualizer.device)
    bounding_boxes = []
    if visualizer.annotations_df is not None:
        image_name = os.path.basename(image_path)
        annotation_row = visualizer.annotations_df[visualizer.annotations_df['image_name'] == image_name]
        if not annotation_row.empty:
            ann_str = annotation_row['annotation'].iloc[0]
            if ann_str.strip() and ann_str.lower() != 'nan':
                bounding_boxes = get_bounding_boxes_from_annotation(ann_str, original_width, original_height)
    results = {}
    pg_results_for_image = {}
    # GradCAM
    if visualizer.gradcam and visualizer.gradcam.hooks:
        heatmap_gc_batch, pred_class_idx_gc = visualizer.gradcam.generate_cam(input_tensor_batch)
        if heatmap_gc_batch is not None and heatmap_gc_batch.size > 0:
            results['gradcam'] = heatmap_gc_batch[0]
            pg_results_for_image['GradCAM'] = pointing_game_accuracy(results['gradcam'], bounding_boxes, original_width, original_height)
    # NormGrad Single Layer
    for method in ['scaling', 'conv1x1', 'conv3x3']:
        if visualizer.normgrad_single and visualizer.normgrad_single.hooks:
            heatmap_ng_s_batch, _ = visualizer.normgrad_single(input_tensor_batch, method=method)
            if heatmap_ng_s_batch is not None and heatmap_ng_s_batch.size > 0:
                results[f'normgrad_single_{method}'] = heatmap_ng_s_batch[0]
                pg_results_for_image[f'NormGrad-{method}-Single'] = pointing_game_accuracy(results[f'normgrad_single_{method}'], bounding_boxes, original_width, original_height)
    # NormGrad Combined Layers
    for method in ['scaling', 'conv1x1', 'conv3x3']:
        if visualizer.normgrad_combined and visualizer.normgrad_combined.hooks:
            heatmap_ng_c_batch, _ = visualizer.normgrad_combined(input_tensor_batch, method=method)
            if heatmap_ng_c_batch is not None and heatmap_ng_c_batch.size > 0:
                results[f'normgrad_combined_{method}'] = heatmap_ng_c_batch[0]
                pg_results_for_image[f'NormGrad-{method}-Combined'] = pointing_game_accuracy(results[f'normgrad_combined_{method}'], bounding_boxes, original_width, original_height)
    return pg_results_for_image

def folder_pg_evaluation(visualizer, image_dir, results_csv_path):
    all_pg_scores = defaultdict(list)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for i, image_name in enumerate(image_files):
        image_path = os.path.join(image_dir, image_name)
        pg_results_one_image = visualize_and_pg(image_path, visualizer, save_path=None)
        if pg_results_one_image:
            for method_title, pg_acc in pg_results_one_image.items():
                all_pg_scores[method_title].append(pg_acc)
    summary_data = []
    for method_title, scores in all_pg_scores.items():
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            num_images = len(scores)
            summary_data.append({
                'XAI Method': method_title,
                'Mean PG Accuracy': mean_score,
                'Std Dev PG Accuracy': std_score,
                'N Images Processed': num_images
            })
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by='Mean PG Accuracy', ascending=False).reset_index(drop=True)
    summary_df.to_csv(results_csv_path, index=False)
    print(summary_df)
    print(f"\nSummary saved to: {results_csv_path}")

# -------------------- XAI Visualizer Class --------------------
class XAIVisualizer:
    def __init__(self, model_path: str, model_name: str, input_size: int, annotations_csv_path: Optional[str] = None):
        self.model_name = model_name
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradcam: Optional[GradCam] = None
        self.normgrad_single: Optional[NormGrad] = None
        self.normgrad_combined: Optional[NormGrad] = None
        self.model: Optional[nn.Module] = None
        self.annotations_df: Optional[pd.DataFrame] = None
        if annotations_csv_path and os.path.exists(annotations_csv_path):
            self.annotations_df = pd.read_csv(annotations_csv_path)
            self.annotations_df['annotation'] = self.annotations_df['annotation'].astype(str)
        self.model = ImageQualityModel(Config())
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()
        gradcam_target = 'model.layer4'
        normgrad_single_target = ['model.layer4']
        normgrad_combined_target = ['model.layer1', 'model.layer2', 'model.layer3', 'model.layer4']
        self.model_args_gradcam = DictAsMember({'name': self.model_name, 'target_layer': gradcam_target})
        self.model_args_normgrad_single = DictAsMember({'name': self.model_name, 'target_layers': normgrad_single_target})
        self.model_args_normgrad_combined = DictAsMember({'name': self.model_name, 'target_layers': normgrad_combined_target})
        self.gradcam = GradCam(self.model, self.model_args_gradcam, self.device, (self.input_size, self.input_size))
        self.normgrad_single = NormGrad(self.model, self.model_args_normgrad_single, (self.input_size, self.input_size))
        self.normgrad_combined = NormGrad(self.model, self.model_args_normgrad_combined, (self.input_size, self.input_size))
    def preprocess_image(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        original_width, original_height = img.size
        resized_display_image = img.resize((self.input_size, self.input_size), Image.LANCZOS)
        transform = get_transforms(is_training=False, input_size=self.input_size)
        input_tensor = transform(img)
        return resized_display_image, input_tensor, (original_width, original_height)
    def __del__(self):
        if hasattr(self, 'gradcam') and self.gradcam: self.gradcam.remove_hooks()
        if hasattr(self, 'normgrad_single') and self.normgrad_single: self.normgrad_single.remove_hooks()
        if hasattr(self, 'normgrad_combined') and self.normgrad_combined: self.normgrad_combined.remove_hooks()

# -------------------- Main Block --------------------
def main():
    Folder = False  # Set to True for folder evaluation, False for single image
    config = Config()
    MODEL_NAME = config.model_config['name']
    INPUT_SIZE = config.model_config['input_size']
    CHECKPOINT_DIR = config.paths['checkpoints']
    RESULTS_DIR = config.paths['results']
    TEST_AFFECTED_DIR = config.paths['test_affected']
    ANNOTATIONS_CSV_PATH = ANNOTATIONS_CSV_PATH
    xai_model_path = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME}_best_acc.pth')
    visualizer = XAIVisualizer(
        model_path=xai_model_path,
        model_name=MODEL_NAME,
        input_size=INPUT_SIZE,
        annotations_csv_path=ANNOTATIONS_CSV_PATH
    )
    if not Folder:
        # Single image evaluation
        image_path = os.path.join(TEST_AFFECTED_DIR, "08001.jpg")  # Example image
        pg_results = visualize_and_pg(image_path, visualizer)
        print("Pointing Game Results for single image:")
        print(pg_results)
    else:
        # Folder evaluation
        results_csv_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_mean_pg_accuracies.csv")
        folder_pg_evaluation(visualizer, TEST_AFFECTED_DIR, results_csv_path)

if __name__ == "__main__":
    main()
