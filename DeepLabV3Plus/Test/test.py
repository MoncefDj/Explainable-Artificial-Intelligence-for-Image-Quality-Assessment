import os
import torch
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch import DeepLabV3Plus
from .config import TEST_CSV, TEST_IMG_DIR, MODEL_CHECKPOINT_PATH, IMG_SIZE, BATCH_SIZE, THRESHOLD

class ObjectCXRSataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, img_size=(256, 256)):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros((3, *self.img_size)), img_name
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        if img.dtype == np.uint8:
            img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        return img, img_name

test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

class SegmentationModel(torch.nn.Module):
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

def predict(model, loader, device, threshold=0.5):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, img_names in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float().cpu().numpy()
            for i, img_name in enumerate(img_names):
                results.append((img_name, preds[i, 0]))
    return results

