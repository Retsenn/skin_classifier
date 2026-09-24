import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.efficientnet_model import AcanthosisEfficientNetB0
from utils.dataset import SkinDataset, get_valid_transforms
from utils.metrics import calculate_clinical_metrics, plot_confusion_matrix, plot_roc_curve

class GradCAM:
    """Generates Class Activation Maps for model interpretability."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, 0]
        score.backward()
        
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0) # ReLU
        if np.max(cam) > 0:
            cam = cam / np.max(cam) # Normalize [0, 1]
            
        cam = cv2.resize(cam, (224, 224))
        return cam, torch.sigmoid(output).item()

def run_evaluation(manifest_csv: str = "processed_dataset/crops/manifest.csv", model_path: str = "results/best_model.pth", save_dir: str = "results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model on device: {device}")
    
    df = pd.read_csv(manifest_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Loaded {len(test_df)} test set samples.")
    
    test_dataset = SkinDataset(test_df, transform=get_valid_transforms())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = AcanthosisEfficientNetB0(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())
            
    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    
    metrics = calculate_clinical_metrics(y_true, y_probs)
    
    print("\n==========================================")
    print("FINAL TEST SET PERFORMANCE METRICS")
    print("==========================================")
    print(f"Accuracy    : {metrics['accuracy']*100:.2f}%")
    print(f"Sensitivity : {metrics['sensitivity']*100:.2f}% (Recall for AN)")
    print(f"Specificity : {metrics['specificity']*100:.2f}% (Recall for Healthy)")
    print(f"Precision   : {metrics['precision']*100:.2f}%")
    print(f"F1-Score    : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC     : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("==========================================\n")
    
    # Plot test metrics
    plots_dir = os.path.join(save_dir, "plots")
    plot_confusion_matrix(metrics["confusion_matrix"], os.path.join(plots_dir, "test_confusion_matrix.png"), title="Test Set Confusion Matrix")
    plot_roc_curve(y_true, y_probs, os.path.join(plots_dir, "test_roc_curve.png"), title="Test Set ROC Curve")
    
    # Save evaluation summary to text
    summary_path = os.path.join(save_dir, "test_evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== EfficientNet-B0 Acanthosis Nigricans Test Evaluation ===\n")
        f.write(f"Accuracy    : {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Sensitivity : {metrics['sensitivity']*100:.2f}%\n")
        f.write(f"Specificity : {metrics['specificity']*100:.2f}%\n")
        f.write(f"Precision   : {metrics['precision']*100:.2f}%\n")
        f.write(f"F1-Score    : {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC     : {metrics['roc_auc']:.4f}\n\n")
        f.write("Confusion Matrix [TN, FP / FN, TP]:\n")
        f.write(f"{metrics['confusion_matrix']}\n")
    print(f"Wrote test evaluation summary to {summary_path}")
    
    # --- Grad-CAM Visualization ---
    gradcam_dir = os.path.join(save_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Target last conv layer in EfficientNet-B0 features
    target_layer = model.backbone.features[8]
    cam_generator = GradCAM(model, target_layer)
    
    # Pick sample images from test set (up to 8 samples: 4 AN, 4 Healthy)
    an_samples = test_df[test_df["label_id"] == 1].head(4)
    healthy_samples = test_df[test_df["label_id"] == 0].head(4)
    samples_df = pd.concat([an_samples, healthy_samples]).reset_index(drop=True)
    
    valid_tf = get_valid_transforms()
    
    for idx, row in samples_df.iterrows():
        img_path = row["filepath"]
        true_label = row["label_name"]
        
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        input_tensor = valid_tf(image=orig_rgb)["image"].unsqueeze(0).to(device)
        input_tensor.requires_grad = True
        
        cam, pred_prob = cam_generator.generate_cam(input_tensor)
        
        # Colorize heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = (0.6 * orig_rgb + 0.4 * heatmap).astype(np.uint8)
        
        pred_label = "AN" if pred_prob >= 0.5 else "Healthy"
        
        # Plot visual comparison
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(orig_rgb)
        plt.title(f"Input ({true_label})")
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap="jet")
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f"Pred: {pred_label} ({pred_prob*100:.1f}%)")
        plt.axis("off")
        
        plt.tight_layout()
        out_fname = f"gradcam_sample_{idx+1:02d}_{true_label}_pred_{pred_label}.png"
        plt.savefig(os.path.join(gradcam_dir, out_fname), dpi=300)
        plt.close()
        
    print(f"Generated Grad-CAM heatmap overlay figures in {gradcam_dir}")

if __name__ == "__main__":
    run_evaluation()
