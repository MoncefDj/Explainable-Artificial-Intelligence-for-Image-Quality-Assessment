import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm  # progress bar
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to your folders
train_dir = "/home/linati/object-CXR_EB0/object-CXR/train"
val_dir   = "/home/linati/object-CXR_EB0/object-CXR/dev"

# Transforms
transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(),
    transforms.RandomAffine(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# Confirm class<->index mapping
print("Train classes:", train_dataset.classes)
print("Train class_to_idx:", train_dataset.class_to_idx)
# Expect: ['affected', 'normal'] → {'affected': 0, 'normal': 1}

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model
# ---------------------------------------------------------------
# NOTE: This EfficientNet-B0 is initialized with ImageNet-pretrained weights
model = models.efficientnet_b0(pretrained=True)
# ---------------------------------------------------------------
# Modify the classifier for binary classification
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)
nn.init.kaiming_normal_(model.classifier[1].weight, nonlinearity='relu')
model = model.to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop with progress bars
best_val_acc = 0.0
num_epochs = 20 # As per original range(1, 21)

# Ensure the directory for saving models exists
output_dir = "." # Save in the current directory (efficientnetb0/)
os.makedirs(output_dir, exist_ok=True)

for epoch in range(1, num_epochs + 1):
    # -- Training --
    model.train()
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs} [Train]", leave=False)
    for imgs, labels in train_bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_bar.set_postfix(loss=loss.item())

    scheduler.step()

    # -- Validation --
    model.eval()
    correct = total = 0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{num_epochs} [Val]  ", leave=False)
    with torch.no_grad():
        for imgs, labels in val_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            val_bar.set_postfix(acc=correct/total)

    val_acc = correct / total
    print(f"Epoch {epoch:02d}/{num_epochs} — Val Acc: {val_acc:.4f}")

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_save_path = os.path.join(output_dir, f"best_efficientnetb0_object_cxr_acc_{best_val_acc:.4f}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved best model to {model_save_path}")

print("Training complete.")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
