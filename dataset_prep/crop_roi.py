import os
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def crop_and_preprocess_roi(manifest_json: str, output_crops_dir: str, target_size: int = 224, margin_ratio: float = 0.15):
    with open(manifest_json, "r") as f:
        records = json.load(f)
        
    os.makedirs(output_crops_dir, exist_ok=True)
    manifest_rows = []
    
    count_success = 0
    count_fail = 0
    
    for r in records:
        img_path = r["file_path"]
        new_split = r["new_split"]
        label_name = r["consolidated_label_name"]
        label_id = r["consolidated_label_id"]
        fname = r["file_name"]
        bbox = r["bbox"] # [x, y, w, h]
        
        target_dir = os.path.join(output_crops_dir, new_split, label_name)
        os.makedirs(target_dir, exist_ok=True)
        out_filepath = os.path.join(target_dir, fname)
        
        if not os.path.exists(img_path):
            count_fail += 1
            continue
            
        try:
            img = cv2.imread(img_path)
            if img is None:
                count_fail += 1
                continue
                
            img_h, img_w = img.shape[:2]
            
            if bbox is not None and len(bbox) == 4:
                x, y, w, h = bbox
                # Add context margin
                dw = w * margin_ratio
                dh = h * margin_ratio
                
                x1 = max(0, int(x - dw))
                y1 = max(0, int(y - dh))
                x2 = min(img_w, int(x + w + dw))
                y2 = min(img_h, int(y + h + dh))
                
                crop = img[y1:y2, x1:x2]
            else:
                crop = img # Fallback to full image if bbox is missing
                
            ch, cw = crop.shape[:2]
            if ch == 0 or cw == 0:
                crop = img
                ch, cw = crop.shape[:2]
                
            # Pad to square aspect ratio
            max_side = max(ch, cw)
            pad_top = (max_side - ch) // 2
            pad_bottom = max_side - ch - pad_top
            pad_left = (max_side - cw) // 2
            pad_right = max_side - cw - pad_left
            
            # Border reflect padding to preserve texture
            padded_crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
            
            # Resize to target resolution (224x224)
            resized_crop = cv2.resize(padded_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            # Save cropped image
            cv2.imwrite(out_filepath, resized_crop)
            
            manifest_rows.append({
                "filepath": out_filepath,
                "file_name": fname,
                "split": new_split,
                "label_name": label_name,
                "label_id": label_id,
                "subject_id": r["subject_id"]
            })
            count_success += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            count_fail += 1
            
    df = pd.DataFrame(manifest_rows)
    manifest_csv_path = os.path.join(output_crops_dir, "manifest.csv")
    df.to_csv(manifest_csv_path, index=False)
    
    print(f"\n--- ROI Cropping Summary ---")
    print(f"Successfully processed: {count_success}")
    print(f"Failed / Missing: {count_fail}")
    print(f"Manifest written to: {manifest_csv_path}")

if __name__ == "__main__":
    crop_and_preprocess_roi(
        manifest_json="processed_dataset/resplit_annotations.json",
        output_crops_dir="processed_dataset/crops",
        target_size=224,
        margin_ratio=0.15
    )
