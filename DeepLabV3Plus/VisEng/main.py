import torch
import os
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
from segmentation_models_pytorch import DeepLabV3Plus
import albumentations as A

from .config import (
    TEST_CSV, TEST_IMG_DIR, MODEL_PATH, IMG_SIZE, DEVICE,
    NORMGRAD_TARGET_LAYERS_LIST, NORMGRAD_EPSILON,
    SIGMOID_K, SIGMOID_THRESH, WEIGHT_IMPORTANCE, WEIGHT_SIZE_PENALTY,
    SALIENCY_FILTER_THRESHOLD, LLM_MODEL_NAME, LLM_MAX_TOKENS, LLM_TEMP
)
from .visualize import plot_analysis_results
from .llm_analysis import format_analysis_for_llm, run_llm_analysis

# --- Minimal Dataset for Inference ---
class ObjectCXRSataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, img_size=(256, 256)):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            return torch.zeros((3, *self.img_size)), torch.zeros((1, *self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        else:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        if img.dtype == np.uint8:
            img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask

def load_model(model_path, device):
    model = DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", classes=1, activation=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    # Load dataset and model
    test_transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])
    test_dataset = ObjectCXRSataset(TEST_CSV, TEST_IMG_DIR, transform=test_transform, img_size=(IMG_SIZE, IMG_SIZE))
    model = load_model(MODEL_PATH, DEVICE)

    # Select image index for analysis
    image_index = 0  # Change as needed
    img_tensor, _ = test_dataset[image_index]
    img_tensor_batch = img_tensor.unsqueeze(0).to(DEVICE)

    # --- Analysis logic (object detection, saliency, scoring) ---
    # Placeholder: Replace with your actual analysis logic
    # For demonstration, we use dummy outputs for the plotting and LLM
    object_scores_data_all = []  # Fill with your object detection/scoring results
    binary_mask = None
    labeled_mask = None
    normgrad_heatmap = None
    overall_image_quality_all = 1.0
    total_penalty_sum_all = 0.0
    num_objects_all = 0
    overall_image_quality_filtered = 1.0
    total_penalty_sum_filtered = 0.0
    num_objects_filtered = 0
    filtered_object_ids = []
    config_params = {
        'normgrad_layer': f"Combined NormGrad O1 ({len(NORMGRAD_TARGET_LAYERS_LIST)} layers)",
        'sigmoid_k': SIGMOID_K,
        'sigmoid_thresh': SIGMOID_THRESH,
        'weight_importance': WEIGHT_IMPORTANCE,
        'weight_size_penalty': WEIGHT_SIZE_PENALTY
    }

    # --- Plotting ---
    plot_analysis_results(
        img_tensor, image_index, object_scores_data_all, binary_mask, labeled_mask, normgrad_heatmap,
        overall_image_quality_all, total_penalty_sum_all, num_objects_all,
        overall_image_quality_filtered, total_penalty_sum_filtered, num_objects_filtered,
        filtered_object_ids, SALIENCY_FILTER_THRESHOLD, config_params
    )

    # --- LLM Analysis ---
    try:
        from gpt4all import GPT4All
        llm = GPT4All(LLM_MODEL_NAME, device=DEVICE, n_ctx=8192)
        prompt = format_analysis_for_llm(
            image_index, config_params, object_scores_data_all, num_objects_all, [],
            overall_image_quality_all, num_objects_filtered, filtered_object_ids,
            overall_image_quality_filtered, SALIENCY_FILTER_THRESHOLD, binary_mask
        )
        response = run_llm_analysis(llm, prompt, max_tokens=LLM_MAX_TOKENS, temp=LLM_TEMP)
        print("\n--- LLM Explanation ---\n", response, "\n-----------------------")
    except ImportError:
        print("gpt4all not installed. Skipping LLM analysis.")

if __name__ == "__main__":
    main()
