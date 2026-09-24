import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.efficientnet_model import AcanthosisEfficientNetB0
from utils.dataset import SkinDataset, get_train_transforms, get_valid_transforms
from utils.metrics import calculate_clinical_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        
        all_probs.extend(probs)
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_clinical_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = epoch_loss
    return metrics

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, targets)
        
        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs).cpu().numpy()
        
        all_probs.extend(probs)
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = calculate_clinical_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = epoch_loss
    return metrics

def run_training(manifest_csv: str = "processed_dataset/crops/manifest.csv", save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv(manifest_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)} | Valid samples: {len(valid_df)}")
    
    # Calculate class weights for BCEWithLogitsLoss
    num_pos = sum(train_df["label_id"] == 1)
    num_neg = sum(train_df["label_id"] == 0)
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32).to(device)
    print(f"Class Imbalance - Healthy: {num_neg}, AN: {num_pos} | Positive Class Weight: {pos_weight.item():.3f}")
    
    train_dataset = SkinDataset(train_df, transform=get_train_transforms())
    valid_dataset = SkinDataset(valid_df, transform=get_valid_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    
    model = AcanthosisEfficientNetB0(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = []
    best_val_auc = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pth")
    
    # --- STAGE 1: Head Warmup (8 Epochs) ---
    print("\n==========================================")
    print("STAGE 1: Warming up classification head")
    print("==========================================")
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    
    for epoch in range(1, 9):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate_epoch(model, valid_loader, criterion, device)
        elapsed = time.time() - t0
        
        print(f"Stage 1 Epoch {epoch:02d}/08 [{elapsed:.1f}s] | "
              f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.4f} AUC: {train_m['roc_auc']:.4f} | "
              f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} Sens: {val_m['sensitivity']:.4f} Spec: {val_m['specificity']:.4f} AUC: {val_m['roc_auc']:.4f}")
        
        history.append({
            "stage": 1, "epoch": epoch,
            "train_loss": train_m["loss"], "train_acc": train_m["accuracy"], "train_auc": train_m["roc_auc"],
            "val_loss": val_m["loss"], "val_acc": val_m["accuracy"], "val_sens": val_m["sensitivity"], "val_spec": val_m["specificity"], "val_auc": val_m["roc_auc"]
        })
        
        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = val_m["roc_auc"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint (Val AUC: {best_val_auc:.4f})")
            
    # --- STAGE 2: Full Fine-Tuning (25 Epochs) ---
    print("\n==========================================")
    print("STAGE 2: Full Backbone Fine-Tuning")
    print("==========================================")
    model.unfreeze_all()
    
    # Differential learning rates for backbone vs head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.features.parameters(), "lr": 1e-4},
        {"params": model.backbone.classifier.parameters(), "lr": 5e-4}
    ], weight_decay=1e-2)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    
    for epoch in range(9, 34):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate_epoch(model, valid_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        print(f"Stage 2 Epoch {epoch:02d}/33 [{elapsed:.1f}s] | "
              f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.4f} AUC: {train_m['roc_auc']:.4f} | "
              f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} Sens: {val_m['sensitivity']:.4f} Spec: {val_m['specificity']:.4f} AUC: {val_m['roc_auc']:.4f}")
        
        history.append({
            "stage": 2, "epoch": epoch,
            "train_loss": train_m["loss"], "train_acc": train_m["accuracy"], "train_auc": train_m["roc_auc"],
            "val_loss": val_m["loss"], "val_acc": val_m["accuracy"], "val_sens": val_m["sensitivity"], "val_spec": val_m["specificity"], "val_auc": val_m["roc_auc"]
        })
        
        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = val_m["roc_auc"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint (Val AUC: {best_val_auc:.4f})")
            
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_df["epoch"], history_df["train_auc"], label="Train ROC-AUC")
    plt.plot(history_df["epoch"], history_df["val_auc"], label="Val ROC-AUC")
    plt.plot(history_df["epoch"], history_df["val_sens"], label="Val Sensitivity", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation ROC-AUC & Sensitivity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "plots", "training_curves.png"), dpi=300)
    plt.close()
    
    print(f"\nTraining Complete! Best Validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {best_model_path}")

if __name__ == "__main__":
    run_training()
