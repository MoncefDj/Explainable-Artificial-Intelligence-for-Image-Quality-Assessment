import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
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
    transforms.ColorJitter(),      # Kept as in original, though typically not in val transform
    transforms.RandomAffine(degrees=10), # Kept as in original, though typically not in val transform
    transforms.RandomHorizontalFlip(), # Kept as in original, though typically not in val transform
    transforms.ToTensor()
])

# Dataset and DataLoader for Validation
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
print("Validation classes:", val_dataset.classes)
print("Validation class_to_idx:", val_dataset.class_to_idx)
# Expect: ['affected', 'normal'] → {'affected': 0, 'normal': 1}

# Model Definition (ResNet34)
# Must match the architecture of the saved model
model = models.resnet34(pretrained=False) # Set pretrained=False as we are loading weights
model.fc = nn.Linear(model.fc.in_features, 2) # Assuming 2 output classes
# Kaiming initialization was in training, not strictly needed here if loading full state_dict

# Load the best saved model weights
model_path = "/home/linati/object-CXR_EB0/resnet34/0.8521478521478522best_resnet34_object_cxr.pth"
print(f"Loading model from: {model_path}")
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded model weights from {model_path}")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {model_path}.")
    exit() # Exit if model not found
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

model.to(device)
# Set the model to evaluation mode
model.eval()

# Variables for calculating accuracy
correct_predictions = 0
total_samples = 0

# Lists to store labels and scores for AUC calculation and other metrics
all_true_labels_list = []
all_pred_scores_list = []
all_predicted_labels_list = []

# Disable gradient calculations
with torch.no_grad():
    # Iterate over the validation data
    val_progress_bar = tqdm(val_loader, desc="Evaluating on Validation Set", leave=False)
    for inputs, actual_labels in val_progress_bar:
        inputs = inputs.to(device)
        actual_labels = actual_labels.to(device)

        # Get model outputs
        outputs = model(inputs)

        # Get predicted class (class with the highest score)
        _, predicted_labels = torch.max(outputs, 1)

        # Update total samples and correct predictions
        total_samples += actual_labels.size(0)
        correct_predictions += (predicted_labels == actual_labels).sum().item()
        
        val_progress_bar.set_postfix(acc=(correct_predictions / total_samples))

        # For AUC: store true labels and predicted scores for the positive class
        # Assuming 'affected' is class 0 and is the positive class for AUC
        probabilities = torch.softmax(outputs, dim=1)
        # Scores for 'affected' class (class 0, based on class_to_idx)
        scores_positive_class = probabilities[:, 0] 

        all_true_labels_list.append(actual_labels.cpu().numpy())
        all_pred_scores_list.append(scores_positive_class.cpu().detach().numpy())
        all_predicted_labels_list.append(predicted_labels.cpu().numpy())

# Calculate validation accuracy
if total_samples > 0:
    validation_accuracy = correct_predictions / total_samples
    print(f"\nFinal Validation Accuracy: {validation_accuracy:.4f} ({correct_predictions}/{total_samples})")
else:
    print("\nNo samples found in validation loader. Cannot calculate accuracy.")
    validation_accuracy = np.nan

# Concatenate all collected labels and scores
y_true_np = np.concatenate(all_true_labels_list) if all_true_labels_list else np.array([])
y_scores_np = np.concatenate(all_pred_scores_list) if all_pred_scores_list else np.array([])
y_pred_np = np.concatenate(all_predicted_labels_list) if all_predicted_labels_list else np.array([])

if y_true_np.size > 0 and y_scores_np.size > 0 :
    # Calculate AUC
    # Transform y_true_np for AUC: 'affected' (class 0) becomes 1, 'normal' (class 1) becomes 0
    # This assumes class_to_idx mapping is {'affected': 0, 'normal': 1}
    y_true_auc = (y_true_np == 0).astype(int) 

    if len(np.unique(y_true_auc)) > 1:
        auc_score = roc_auc_score(y_true_auc, y_scores_np)
        print(f"Validation AUC (for 'affected' class as positive): {auc_score:.4f}")
    else:
        print("AUC not calculated: Only one class present in y_true_auc.")
        auc_score = np.nan
else:
    print("Not enough data to calculate AUC.")
    auc_score = np.nan


if y_true_np.size > 0 and y_pred_np.size > 0:
    # Define class names based on the assumed mapping {'affected': 0, 'normal': 1}
    class_names = ['affected (0)', 'normal (1)'] # From val_dataset.class_to_idx

    # Calculate Precision, Recall, F1-Score
    print("\nClassification Report:")
    report = classification_report(y_true_np, y_pred_np, target_names=class_names, zero_division=0)
    print(report)

    # Plot Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_true_np, y_pred_np)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for resnet34 ({os.path.basename(model_path)})')
    
    # Save or show the plot
    try:
        cm_filename = f"confusion_matrix_resnet34_{os.path.basename(model_path).replace('.pth', '')}.png"
        plt.savefig(cm_filename)
        print(f"Confusion matrix saved as {cm_filename}")
        # plt.show() # Uncomment to display plot
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")
else:
    print("\nNot enough data to calculate classification report or confusion matrix.")

print("\nEvaluation complete.")
