# app/services/data_loader_service.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset
from config import IMG_SIZE, TEST_CSV_NAME # Import necessary items from main config

def parse_annotations(ann_str, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if not isinstance(ann_str, str) or ann_str.strip() == "": return mask
    h, w = img_shape[:2]
    try:
        objects = ann_str.split(';')
        for obj_item in objects:
            if not obj_item.strip(): continue
            parts = list(map(int, obj_item.split()))
            obj_type, coords = parts[0], parts[1:]
            if obj_type == 0 and len(coords) == 4:
                x1,y1,x2,y2 = map(lambda v:max(0,v), coords); x2,y2=min(x2,w),min(y2,h)
                if x1<x2 and y1<y2: mask[y1:y2,x1:x2]=1
            elif obj_type == 1 and len(coords) == 4:
                x1,y1,x2,y2 = map(lambda v:max(0,v), coords); x2,y2=min(x2,w),min(y2,h)
                if x1<x2 and y1<y2:
                    center,axes = ((x1+x2)//2,(y1+y2)//2),((x2-x1)//2,(y2-y1)//2)
                    if axes[0]>0 and axes[1]>0: cv2.ellipse(mask,center,axes,0,0,360,1,-1)
                    elif axes[0]==0 and axes[1]>0: cv2.line(mask,(center[0],y1),(center[0],y2),1,1)
                    elif axes[1]==0 and axes[0]>0: cv2.line(mask,(x1,center[1]),(x2,center[1]),1,1)
            elif obj_type == 2 and len(coords)>=6 and len(coords)%2==0:
                points=np.array(coords).reshape(-1,2)
                points[:,0]=np.clip(points[:,0],0,w-1); points[:,1]=np.clip(points[:,1],0,h-1)
                cv2.fillPoly(mask,[points],1)
    except Exception as e: print(f"DataLoader Service: Error parsing annotation: {e}")
    return mask

class ObjectCXRSataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, img_size=(IMG_SIZE,IMG_SIZE)):
        self.data = pd.DataFrame()
        try:
            self.data = pd.read_csv(csv_file)
            if self.data.empty: print(f"DataLoader Service Warning: CSV {csv_file} is empty.")
        except Exception as e:
            print(f"DataLoader Service Error reading CSV {csv_file}: {e}"); self.data = pd.DataFrame()
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size # Tuple (height, width) for consistency with A.Resize

    def __len__(self): return len(self.data) if hasattr(self.data, 'empty') and not self.data.empty else 0
    
    def get_image_path(self, idx): # Helper
        if not (0 <= idx < len(self.data)): return None
        img_name = self.data.iloc[idx,0]
        base_img_dir_name = os.path.splitext(TEST_CSV_NAME)[0]
        
        if os.path.basename(self.img_dir) == base_img_dir_name: # img_dir is .../dev
             path_candidate = os.path.join(self.img_dir, img_name)
             if os.path.exists(path_candidate): return path_candidate
        
        path_candidate_with_subdir = os.path.join(self.img_dir, base_img_dir_name, img_name) # img_dir is DATA_BASE_PATH
        if os.path.exists(path_candidate_with_subdir): return path_candidate_with_subdir

        path_candidate_flat = os.path.join(self.img_dir, img_name) # img_dir is DATA_BASE_PATH, flat structure
        if os.path.exists(path_candidate_flat): return path_candidate_flat

        print(f"DataLoader Service Warning: Image path for {img_name} (idx {idx}) not found with common structures under {self.img_dir}.")
        return os.path.join(self.img_dir, img_name) # Fallback

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.data)):
            print(f"DataLoader Service Critical Error: Index {idx} out of range for {len(self.data)}")
            return (torch.zeros((3, self.img_size[0], self.img_size[1]),dtype=torch.float32), None, None)
        
        img_name = self.data.iloc[idx,0]
        ann_str = self.data.iloc[idx,1] if self.data.shape[1]>1 else ""
        img_path = self.get_image_path(idx)
        original_image_cv = None
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"DataLoader Service Warning: Fail load {img_path}. Dummy.")
                return (torch.zeros((3, self.img_size[0], self.img_size[1]),dtype=torch.float32), None, img_path)

            original_image_cv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
            img_rgb_for_transform = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # Mask data parsing is not directly used for tensor output but good to keep if needed
            # mask_data=parse_annotations(ann_str,img_rgb_for_transform.shape) 

            img_transformed_for_analysis = None
            if self.transform:
                augmented = self.transform(image=img_rgb_for_transform) # Only image for tensor
                img_transformed_for_analysis = augmented['image']
            else:
                img_transformed_for_analysis = cv2.resize(img_rgb_for_transform, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_AREA) # CV2 uses (W,H)

            if img_transformed_for_analysis.dtype==np.uint8:
                img_transformed_for_analysis=img_transformed_for_analysis/255.0
            
            analysis_tensor = torch.tensor(img_transformed_for_analysis,dtype=torch.float32).permute(2,0,1)
            return analysis_tensor, original_image_cv, img_path

        except Exception as e:
            print(f"DataLoader Service Error item {idx} ({img_name}): {e}")
            import traceback
            traceback.print_exc()
            return (torch.zeros((3, self.img_size[0], self.img_size[1]),dtype=torch.float32), None, img_path)

# Global transform object for consistency, used by DataLoader instances
# Albumentations Resize takes (height, width)
test_transform_global = A.Compose([A.Resize(height=IMG_SIZE, width=IMG_SIZE)])


class DataLoaderService:
    def __init__(self, data_base_path, csv_name, img_size_tuple): # img_size_tuple (height, width)
        self.data_base_path = data_base_path
        self.csv_name = csv_name
        self.img_size_tuple = img_size_tuple # For A.Resize (H, W)
        self.dataset = None
        self.max_index = -1
        self.dataset_message = "Dataset not yet loaded."
        self.dataset_ok = False
        self._load_dataset()

    def _load_dataset(self):
        csv_path = os.path.join(self.data_base_path, self.csv_name)
        img_dir_name_from_csv = os.path.splitext(self.csv_name)[0]
        
        img_dir = self.data_base_path # Default
        potential_img_subdir = os.path.join(self.data_base_path, img_dir_name_from_csv)
        if os.path.isdir(potential_img_subdir):
            img_dir = potential_img_subdir
        elif os.path.isdir(os.path.join(self.data_base_path, "images")):
            img_dir = os.path.join(self.data_base_path, "images")
        
        print(f"DataLoader Service: Determined image directory: {img_dir}")

        if not os.path.exists(csv_path):
            self.dataset_message = f"Error: Dataset CSV not found at {csv_path}"
            self.dataset_ok = False; return
        if not os.path.isdir(img_dir):
            self.dataset_message = f"Error: Image directory not found at {img_dir}"
            self.dataset_ok = False; return

        self.dataset = ObjectCXRSataset(csv_path, img_dir, transform=test_transform_global, img_size=self.img_size_tuple)
        if self.dataset and len(self.dataset) > 0:
            self.max_index = len(self.dataset) - 1
            self.dataset_message = f"Dataset loaded successfully with {len(self.dataset)} images."
            self.dataset_ok = True
            print(f"DataLoader Service: {self.dataset_message} Max index: {self.max_index}")
        else:
            self.dataset_message = "Warning: Dataset is empty or failed to load."
            self.max_index = -1; self.dataset_ok = False
            print(f"DataLoader Service: {self.dataset_message}")
            
    def get_image_by_index(self, index):
        if not self.dataset_ok or not (0 <= index <= self.max_index):
            print(f"DataLoader Service: Invalid index {index} or dataset not OK.")
            return None, None, None 
        return self.dataset[index] 

    def get_image_by_path(self, image_path_param, transform_for_tensor=True):
        if not os.path.exists(image_path_param):
            print(f"DataLoader Service: Image not found at path: {image_path_param}")
            return None, None 
        try:
            img_cv_bgr = cv2.imread(image_path_param)
            if img_cv_bgr is None:
                print(f"DataLoader Service: Failed to read image at {image_path_param}")
                return None, None
            
            original_cv_image = cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB)
            analysis_tensor = None

            if transform_for_tensor:
                img_rgb_for_transform = original_cv_image.copy() 
                # Use the global transform which is A.Resize(IMG_SIZE, IMG_SIZE)
                augmented = test_transform_global(image=img_rgb_for_transform)
                img_transformed = augmented['image']
                
                if img_transformed.dtype == np.uint8:
                    img_transformed = img_transformed / 255.0
                analysis_tensor = torch.tensor(img_transformed, dtype=torch.float32).permute(2, 0, 1)
            
            return analysis_tensor, original_cv_image
        except Exception as e:
            print(f"DataLoader Service Error loading image by path {image_path_param}: {e}")
            return None, None