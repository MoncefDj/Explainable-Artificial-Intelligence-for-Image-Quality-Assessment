### You can download the Dataset from Kaggle website:

- [Foreign objects in chest X-rays](https://www.kaggle.com/datasets/raddar/foreign-objects-in-chest-xrays/data) by raddar.

or from Kaggel `API`:

1. First download `kagglehub`:
```bash
pip install kagglehub
```
2. Then run this code:
```python
import kagglehub
def download_and_move_dataset(dataset_name, destination_path):
    """
    Download a Kaggle dataset and move contents to the destination path.

    Args:
        dataset_name (str): The Kaggle dataset identifier
        destination_path (str): Where to move the dataset files
    """
    try:
        # Download the dataset
        print(f"Downloading dataset: {dataset_name}...")
        download_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {os.path.abspath(download_path)}")

        # Ensure destination directory exists
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
            print(f"Created destination directory: {destination_path}")

        # Move all content
        print(f"\033[92mMoving the dataset to{destination_path}!\033[0m")

        count = 0
        for item in os.listdir(download_path):
            source_item = os.path.join(download_path, item)
            dest_item = os.path.join(destination_path, item)

            if os.path.exists(dest_item):
                print(f"Warning: {dest_item} already exists. Skipping.")
                continue

            try:
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item)
                else:
                    shutil.copy2(source_item, dest_item)
                count += 1
                print(f"Copied: {item}")
            except Exception as e:
                print(f"Error processing {item}: {e}")

        print(f"Successfully transferred {count} items to {destination_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTroubleshooting steps:")
        print("1. Update kagglehub: !pip install --upgrade kagglehub")
        print("2. Verify you have the right dataset name")
        print("3. Make sure you're authenticated with Kaggle")
        print("   !pip install kaggle")
        print("   Upload your kaggle.json file and run:")
        print("   !mkdir -p ~/.kaggle")
        print("   !cp kaggle.json ~/.kaggle/")
        print("   !chmod 600 ~/.kaggle/kaggle.json")

# Example usage
dataset_name = "raddar/foreign-objects-in-chest-xrays"
destination_path = "."
download_and_move_dataset(dataset_name, destination_path)
```

> PS: This dataset not separated into `normal` and `affected`.

### To separate the data run this code 
```python
def sort_images_by_anno(csv_path, images_root, normal_folder, affected_folder):
    """Sorts images into 'normal' and 'affected' based on CSV annotation."""
    if not os.path.exists(csv_path):
        print(f"Error: Training CSV not found at {csv_path}. Cannot sort images.")
        return False
    if not os.path.exists(images_root):
        print(f"Error: Image root directory not found at {images_root}. Cannot sort images.")
        return False

    # Load CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return False

    os.makedirs(normal_folder, exist_ok=True)
    os.makedirs(affected_folder, exist_ok=True)

    moved_count = 0
    skipped_count = 0
    missing_count = 0
    print(f"Sorting images from {images_root} based on {csv_path}...")

    # Check expected columns
    if 'image_name' not in df.columns or 'annotation' not in df.columns:
        print(f"Error: CSV must contain 'image_name' and 'annotation' columns.")
        return False

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sorting Images"):
        image_name = row['image_name']
        # Handle potential NaN annotations gracefully
        anno = str(row['annotation']).strip().lower()
        source_path = os.path.join(images_root, image_name)

        if not os.path.exists(source_path):
            missing_count += 1
            continue

        # Determine destination folder
        if anno == "" or anno == "nan":
            dest_path = os.path.join(normal_folder, image_name)
            target_folder = normal_folder
        else:
            dest_path = os.path.join(affected_folder, image_name)
            target_folder = affected_folder

        # Move the file if not already in the correct place
        if not os.path.exists(dest_path):
            try:
                shutil.move(source_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {source_path} to {dest_path}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1
            if os.path.exists(source_path):
                 os.remove(source_path)


    print(f"Sorting complete. Moved: {moved_count}, Skipped/Already Sorted: {skipped_count}, Missing Source: {missing_count}")
    return True
```
