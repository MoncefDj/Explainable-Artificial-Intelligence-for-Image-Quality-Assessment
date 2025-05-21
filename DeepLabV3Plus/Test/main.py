import torch
from torch.utils.data import DataLoader
from .config import TEST_CSV, TEST_IMG_DIR, MODEL_CHECKPOINT_PATH, IMG_SIZE, BATCH_SIZE, THRESHOLD
from .test import ObjectCXRSataset, SegmentationModel, test_transform, predict

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = ObjectCXRSataset(TEST_CSV, TEST_IMG_DIR, transform=test_transform, img_size=(IMG_SIZE, IMG_SIZE))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0)
    model = SegmentationModel()
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
    model.to(device)
    results = predict(model, test_loader, device, threshold=THRESHOLD)
    # Example: save results as needed
    # import pickle; pickle.dump(results, open("test_predictions.pkl", "wb"))

if __name__ == "__main__":
    main()
