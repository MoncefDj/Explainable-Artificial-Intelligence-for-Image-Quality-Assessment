import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths to your folders (only val_dir needed for testing)
val_dir   = "/home/linati/object-CXR_EB0/object-CXR/dev"

# Transforms (must be same as used in training for validation set)
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(), # Kept as in original, though typically not in val transform
    transforms.RandomAffine(degrees=10), # Kept as in original, though typically not in val transform
    transforms.RandomHorizontalFlip(), # Kept as in original, though typically not in val transform
    transforms.ToTensor()
])

# Dataset and DataLoader for Validation
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
print("Validation classes:", val_dataset.classes)
print("Validation class_to_idx:", val_dataset.class_to_idx)


# Model Definition (EfficientNet-B0)
# Must match the architecture of the saved model
model = models.efficientnet_b0(pretrained=False) # Set pretrained=False as we are loading weights
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2) # Assuming 2 classes: 'affected', 'normal'


# Load the model
# Note: The filename implies the model achieved a certain accuracy.
# Ensure this path is correct and accessible.
model_path = "/home/linati/object-CXR_EB0/efficientnetb0/best_efficientnetb0_object_cxr_acc_0.8671.pth"
print(f"Loading model from: {model_path}")

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model weights from {model_path}")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {model_path}.")
    exit() # Exit if model not found, as evaluation is not possible
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

model.to(device)
model.eval()

all_labels = []
all_preds_probs_class0 = [] # Probabilities for class 0 ('affected')
all_preds_classes = []

# No need for gradients during evaluation
with torch.no_grad():
    val_bar_eval = tqdm(val_loader, desc="Evaluating Model", leave=False)
    for imgs, labels in val_bar_eval:
        imgs, labels = imgs.to(device), labels.to(device)
        
        outputs = model(imgs)
        
        # Probabilities for each class
        probs = F.softmax(outputs, dim=1)
        
        # Store true labels
        all_labels.extend(labels.cpu().numpy())
        
        # Store probabilities for class 0 (assumed 'affected') for AUC
        # Based on class_to_idx: {'affected': 0, 'normal': 1}
        all_preds_probs_class0.extend(probs[:, 0].cpu().numpy())
        
        # Get predicted classes (0 or 1)
        _, predicted_classes = torch.max(outputs, 1)
        all_preds_classes.extend(predicted_classes.cpu().numpy())

# Convert lists to numpy arrays for metrics calculation
all_labels_np = np.array(all_labels)
all_preds_probs_class0_np = np.array(all_preds_probs_class0)
all_preds_classes_np = np.array(all_preds_classes)
total_samples_eval = len(all_labels_np) # Renamed to avoid conflict if train script variables were in same scope
print(f"Total samples evaluated: {total_samples_eval}")
print(f"\nModel Evaluation Results for: {model_path}")

if total_samples_eval > 0:
    # Calculate Accuracy
    correct_predictions_eval = np.sum(all_preds_classes_np == all_labels_np)
    accuracy = correct_predictions_eval / total_samples_eval
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions_eval}/{total_samples_eval})")
else:
    print("No samples found in validation loader. Cannot calculate accuracy.")
    accuracy = 0.0 # or np.nan


# Calculate AUC without sklearn (original function)
def calculate_auc(y_true, y_scores):
    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # For ROC AUC, we need y_true where 1 is 'affected' (positive class), 0 is 'normal'
    # Original labels are 0 for 'affected', 1 for 'normal'
    y_true_for_auc = 1 - y_true  # Invert labels: 0->1 (affected), 1->0 (normal)
    
    # Sort predictions by score (scores are for the 'affected' class, which is now labeled 1)
    sorted_indices = np.argsort(y_scores)[::-1]  # descending order of scores for 'affected'
    sorted_y_true = y_true_for_auc[sorted_indices]
    
    # Calculate true positive rate (TPR) and false positive rate (FPR) at different thresholds
    n_pos = np.sum(sorted_y_true) # Number of actual 'affected' samples
    n_neg = len(sorted_y_true) - n_pos # Number of actual 'normal' samples
    
    # Edge case: if all examples are from one class
    if n_pos == 0 or n_neg == 0:
        print("AUC calculation: All samples belong to one class. AUC is 0.5 or undefined.")
        return 0.5
    
    # Calculate TPR and FPR for each threshold
    tpr_calc = np.cumsum(sorted_y_true) / n_pos # Renamed to avoid conflict
    fpr_calc = np.cumsum(1 - sorted_y_true) / n_neg # 1 - sorted_y_true gives 1 for false positives
    
    # Add (0,0) and (1,1) points to complete the curve
    tpr_calc = np.concatenate([[0], tpr_calc, [1]])
    fpr_calc = np.concatenate([[0], fpr_calc, [1]])
    
    # Calculate AUC using trapezoidal rule
    # Ensure fpr_calc is monotonically increasing to avoid issues with np.diff
    fpr_calc, unique_indices = np.unique(fpr_calc, return_index=True)
    tpr_calc = tpr_calc[unique_indices]

    if len(fpr_calc) < 2: # Not enough points to form a trapezoid
        print("AUC calculation: Not enough distinct FPR values to calculate AUC. Returning 0.5.")
        return 0.5

    width = np.diff(fpr_calc)
    height = (tpr_calc[1:] + tpr_calc[:-1]) / 2
    auc = np.sum(width * height)
    
    return auc

if all_labels_np.size > 0 and all_preds_probs_class0_np.size > 0:
    # Ensure there's more than one class present in y_true_for_auc for meaningful AUC
    if len(np.unique(1 - all_labels_np)) > 1:
        auc_score = calculate_auc(all_labels_np, all_preds_probs_class0_np)
        print(f"AUC (for 'affected' class as positive, custom calc): {auc_score:.4f}")
    else:
        print("AUC not calculated (custom calc): Only one class present in true labels after transformation.")
        auc_score = np.nan 
else:
    print("Not enough data to calculate AUC (custom calc).")
    auc_score = np.nan


if all_labels_np.size > 0 and all_preds_classes_np.size > 0:
    # --- Metrics Calculation using sklearn ---
    # Define class names based on the assumed mapping: 0 for 'affected', 1 for 'normal'
    class_names = ['affected (0)', 'normal (1)'] # From val_dataset.class_to_idx

    # Calculate Precision, Recall, F1-Score
    print("\nClassification Report (sklearn):")
    report = classification_report(all_labels_np, all_preds_classes_np, target_names=class_names, zero_division=0)
    print(report)

    # Plot Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(all_labels_np, all_preds_classes_np)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {os.path.basename(model_path)}') 
    
    # Save or show the plot
    try:
        cm_filename = f"confusion_matrix_{os.path.basename(model_path).replace('.pth', '')}.png"
        plt.savefig(cm_filename)
        print(f"Confusion matrix saved as {cm_filename}")
        # plt.show() # Uncomment to display plot
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")
else:
    print("\nNot enough data to calculate classification report or confusion matrix.")

print("\nEvaluation complete.")
