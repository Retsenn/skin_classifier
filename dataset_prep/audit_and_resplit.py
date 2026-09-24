import os
import json
import re
from collections import defaultdict
import numpy as np

def extract_subject_id(filename: str) -> str:
    """Extracts a unique subject/patient identifier from filename to prevent data leakage."""
    # Strip Roboflow hash prefix (.rf.[hash].jpg)
    base = re.sub(r"\.rf\.[A-Za-z0-9]+\.(jpg|jpeg|png)$", "", filename, flags=re.IGNORECASE)
    base = re.sub(r"(_jpg|_jpeg|_png)$", "", base, flags=re.IGNORECASE)
    
    # Strip orientation prefixes
    base = re.sub(r"^FLIP_(LEFT|RIGHT)", "", base, flags=re.IGNORECASE)
    
    # Standardize burst photo series
    base = re.sub(r"^images\s*\(\d+\)", "burst_series_1", base, flags=re.IGNORECASE)
    base = re.sub(r"^images\s*-\s*\d{4}-\d{2}-\d{2}T\d+-\d+", "burst_series_2", base, flags=re.IGNORECASE)
    
    # Strip pose/side suffixes (L3, R3, RR4, R, L)
    base = re.sub(r"(R3|RR4|R|L3|L|R)$", "", base)
    
    # Clean trailing underscores/hyphens
    base = base.strip("_-")
    return base if base else filename

def audit_and_resplit(base_dir: str, output_path: str):
    splits = ["train", "valid", "test"]
    all_records = []
    
    for split in splits:
        ann_path = os.path.join(base_dir, split, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            continue
            
        with open(ann_path, "r") as f:
            coco_data = json.load(f)
            
        categories = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
        
        # Map image_id -> list of annotations
        ann_by_img = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            ann_by_img[ann["image_id"]].append(ann)
            
        for img in coco_data.get("images", []):
            img_anns = ann_by_img[img["id"]]
            ann_names = [categories.get(a["category_id"], "") for a in img_anns]
            
            # Determine class label
            if "Acanthosis Nigricans" in ann_names:
                label_name = "AN"
                label_id = 1
            elif "Healthy" in ann_names:
                label_name = "Healthy"
                label_id = 0
            else:
                # Default fallback if category is listed as Neck or unlabelled
                label_name = "Healthy"
                label_id = 0
                
            # Find primary bounding box (prefer AN or Healthy bbox over generic Neck bbox)
            selected_bbox = None
            for a in img_anns:
                cname = categories.get(a["category_id"], "")
                if cname in ["Acanthosis Nigricans", "Healthy"]:
                    selected_bbox = a.get("bbox")
                    break
            if selected_bbox is None and img_anns:
                selected_bbox = img_anns[0].get("bbox")
                
            subject_id = extract_subject_id(img["file_name"])
            
            all_records.append({
                "original_split": split,
                "file_name": img["file_name"],
                "file_path": os.path.join(base_dir, split, img["file_name"]),
                "image_id": img["id"],
                "width": img["width"],
                "height": img["height"],
                "subject_id": subject_id,
                "label_name": label_name,
                "label_id": label_id,
                "bbox": selected_bbox
            })
            
    print(f"Total images collected across splits: {len(all_records)}")
    
    # Group records by subject_id to ensure consistent labeling & split assignment
    subject_to_records = defaultdict(list)
    for r in all_records:
        subject_to_records[r["subject_id"]].append(r)
        
    print(f"Total unique subjects identified: {len(subject_to_records)}")
    
    # Consolidate class label per subject (majority or AN priority)
    subjects = []
    subject_labels = []
    
    for subj_id, recs in subject_to_records.items():
        an_count = sum(1 for r in recs if r["label_id"] == 1)
        healthy_count = sum(1 for r in recs if r["label_id"] == 0)
        
        # If any image of subject shows AN, classify subject as AN (1)
        final_label = 1 if an_count > 0 else 0
        
        for r in recs:
            r["consolidated_label_id"] = final_label
            r["consolidated_label_name"] = "AN" if final_label == 1 else "Healthy"
            
        subjects.append(subj_id)
        subject_labels.append(final_label)
        
    # Perform Group-Stratified Split: 80% Train, 10% Valid, 10% Test
    np.random.seed(42)
    
    # Split subjects into Train (80%) vs Temp (20%)
    an_subjects = [s for s, l in zip(subjects, subject_labels) if l == 1]
    healthy_subjects = [s for s, l in zip(subjects, subject_labels) if l == 0]
    
    np.random.shuffle(an_subjects)
    np.random.shuffle(healthy_subjects)
    
    def split_list(lst, train_ratio=0.8, val_ratio=0.1):
        n = len(lst)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return set(lst[:n_train]), set(lst[n_train:n_train+n_val]), set(lst[n_train+n_val:])
        
    an_train, an_val, an_test = split_list(an_subjects)
    h_train, h_val, h_test = split_list(healthy_subjects)
    
    train_subjects = an_train | h_train
    val_subjects = an_val | h_val
    test_subjects = an_test | h_test
    
    # Assign new_split to every record
    resplit_records = []
    split_counts = defaultdict(lambda: defaultdict(int))
    
    for r in all_records:
        s = r["subject_id"]
        if s in train_subjects:
            new_split = "train"
        elif s in val_subjects:
            new_split = "valid"
        else:
            new_split = "test"
            
        r["new_split"] = new_split
        resplit_records.append(r)
        split_counts[new_split][r["consolidated_label_name"]] += 1
        
    print("\n--- Leak-Free Resplit Summary ---")
    for sp in ["train", "valid", "test"]:
        counts = split_counts[sp]
        print(f"Split [{sp}]: AN={counts['AN']}, Healthy={counts['Healthy']}, Total={counts['AN'] + counts['Healthy']}")
        
    # Check subject overlap
    tr_sub = {r["subject_id"] for r in resplit_records if r["new_split"] == "train"}
    val_sub = {r["subject_id"] for r in resplit_records if r["new_split"] == "valid"}
    te_sub = {r["subject_id"] for r in resplit_records if r["new_split"] == "test"}
    
    print(f"\nSubject Overlap Check:")
    print(f"  Train ∩ Valid: {len(tr_sub & val_sub)}")
    print(f"  Train ∩ Test : {len(tr_sub & te_sub)}")
    print(f"  Valid ∩ Test : {len(val_sub & te_sub)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(resplit_records, f, indent=2)
        
    print(f"\nWrote manifest to {output_path}")

if __name__ == "__main__":
    audit_and_resplit("Acanthosis Nigricans-Forked on 9-22-2026.coco", "processed_dataset/resplit_annotations.json")
