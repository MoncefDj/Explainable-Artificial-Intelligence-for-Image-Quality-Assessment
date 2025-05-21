import torch
from torch.utils.data import DataLoader
from .config import (
    TRAIN_CSV, TRAIN_IMG_DIR, TEST_CSV, TEST_IMG_DIR, IMG_SIZE, N_EPOCHS, BATCH_SIZE,
    LEARNING_RATE, MODEL_SAVE_DIR
)
from .train import (
    ObjectCXRSataset, SegmentationModel, focal_tversky_loss, train_segmentation_model, transform
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ObjectCXRSataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=transform, img_size=(IMG_SIZE, IMG_SIZE))
    val_dataset = ObjectCXRSataset(TEST_CSV, TEST_IMG_DIR, transform=transform, img_size=(IMG_SIZE, IMG_SIZE))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2 if torch.cuda.is_available() else 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0)
    model = SegmentationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    data_loaders = {'train': train_loader, 'val': val_loader}
    train_segmentation_model(model, optimizer, focal_tversky_loss, data_loaders, device, N_EPOCHS, MODEL_SAVE_DIR)

if __name__ == "__main__":
    main()
