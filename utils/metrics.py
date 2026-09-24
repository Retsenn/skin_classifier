import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def calculate_clinical_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
        
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        auc = 0.5
        
    return {
        "accuracy": acc,
        "sensitivity": rec, # Recall
        "specificity": specificity,
        "precision": prec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm: np.ndarray, save_path: str, title: str = "Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Healthy (0)", "AN (1)"], 
                yticklabels=["Healthy (0)", "AN (1)"])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_pred_probs: np.ndarray, save_path: str, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    try:
        auc_val = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        auc_val = 0.5
        
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
