import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.08, 0.08), rotate=(-20, 20), p=0.5, mode=cv2.BORDER_REFLECT),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_valid_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = float(row["label_id"])
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
        return image, torch.tensor(label, dtype=torch.float32)
