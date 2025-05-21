import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch import DeepLabV3Plus
from tqdm import tqdm
from .config import (
    TRAIN_CSV, TRAIN_IMG_DIR, TEST_CSV, TEST_IMG_DIR, IMG_SIZE, N_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, MODEL_SAVE_DIR, NORMGRAD_EPSILON
)

# --- Dataset ---
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
            obj_type = parts[0]
            coords = parts[1:]
            if obj_type == 0 and len(coords) == 4:  # Rectangle
                x1, y1, x2, y2 = map(lambda v: max(0, v), coords)
                x2, y2 = min(x2, w), min(y2, h)
                if x1 < x2 and y1 < y2: mask[y1:y2, x1:x2] = 1
            elif obj_type == 1 and len(coords) == 4:  # Ellipse
                x1, y1, x2, y2 = map(lambda v: max(0, v), coords)
                x2, y2 = min(x2, w), min(y2, h)
                if x1 < x2 and y1 < y2:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                    if axes[0] > 0 and axes[1] > 0:
                        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
            elif obj_type == 2 and len(coords) >= 6 and len(coords) % 2 == 0:  # Polygon
                points = np.array(coords).reshape(-1, 2)
                points[:, 0] = np.clip(points[:, 0], 0, w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, h - 1)
                cv2.fillPoly(mask, [points], 1)
    except Exception as e:
        print(f"Error parsing annotation string '{ann_str}': {e}")
    return mask

class ObjectCXRSataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, img_size=(256, 256)):
        try:
            self.data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}")
            self.data = pd.DataFrame()
            return
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        ann_str = self.data.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros((3, *self.img_size)), torch.zeros((1, *self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = parse_annotations(ann_str, img.shape)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        else:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        if img.dtype == np.uint8:
            img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask

# --- Transforms ---
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# --- Model ---
class SegmentationModel(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", classes=1):
        super().__init__()
        self.model = DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=None
        )
    def forward(self, x):
        return self.model(x)

# --- Loss Function (example: focal tversky loss) ---
def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(pred.size(0), pred.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1)
    true_pos = (pred * target).sum(dim=2)
    false_neg = ((1 - pred) * target).sum(dim=2)
    false_pos = (pred * (1 - target)).sum(dim=2)
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return (1. - tversky_index).mean()

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.):
    t_loss = tversky_loss(pred, target, alpha, beta, smooth)
    return torch.pow(t_loss, gamma)

# --- Evaluation ---
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0
    num_samples = 0
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            output = model(img)
            loss = loss_fn(output, mask)
            total_loss += loss.item() * img.size(0)
            num_samples += img.size(0)
    return total_loss / num_samples if num_samples > 0 else 0

# --- Training Loop ---
def train_segmentation_model(model, optimizer, loss_function, data_loaders, device, epochs, save_dir):
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_wts = None
    best_model_save_path = None

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        num_train_samples = 0
        for inputs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            num_train_samples += inputs.size(0)
        epoch_train_loss = running_train_loss / num_train_samples if num_train_samples > 0 else 0
        history['train_loss'].append(epoch_train_loss)
        epoch_val_loss = evaluate(model, val_loader, device, loss_function)
        history['val_loss'].append(epoch_val_loss)
        if save_dir:
            epoch_save_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_save_path)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    if best_model_wts is not None and save_dir:
        best_model_save_path = os.path.join(save_dir, "best_model.pth")
        torch.save(best_model_wts, best_model_save_path)
    return history, best_model_save_path
