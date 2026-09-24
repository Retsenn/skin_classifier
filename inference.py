import os
import argparse
import cv2
import numpy as np
import torch

from models.efficientnet_model import AcanthosisEfficientNetB0
from utils.dataset import get_valid_transforms

class AcanthosisClassifier:
    def __init__(self, model_path: str = "results/best_model.pth", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading classifier on device: {self.device}")
        self.model = AcanthosisEfficientNetB0(pretrained=False).to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = get_valid_transforms()
        
    def preprocess_image(self, image_path: str, bbox: list = None, margin_ratio: float = 0.15) -> torch.Tensor:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        img_h, img_w = img.shape[:2]
        
        if bbox is not None and len(bbox) == 4:
            x, y, w, h = bbox
            dw = w * margin_ratio
            dh = h * margin_ratio
            x1 = max(0, int(x - dw))
            y1 = max(0, int(y - dh))
            x2 = min(img_w, int(x + w + dw))
            y2 = min(img_h, int(y + h + dh))
            crop = img[y1:y2, x1:x2]
        else:
            crop = img
            
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
        
        padded_crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        resized_crop = cv2.resize(padded_crop, (224, 224), interpolation=cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
        
        tensor = self.transform(image=rgb_img)["image"].unsqueeze(0)
        return tensor.to(self.device)
        
    @torch.no_grad()
    def predict(self, image_path: str, bbox: list = None) -> dict:
        tensor = self.preprocess_image(image_path, bbox=bbox)
        logit = self.model(tensor).squeeze(1)
        prob = torch.sigmoid(logit).item()
        
        predicted_class = "Acanthosis Nigricans" if prob >= 0.5 else "Healthy"
        confidence = prob if prob >= 0.5 else (1.0 - prob)
        
        return {
            "prediction": predicted_class,
            "probability_an": float(prob),
            "confidence": float(confidence * 100.0),
            "image_path": image_path
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Acanthosis Nigricans image classification")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--bbox", type=float, nargs=4, default=None, help="Optional bounding box: x y w h")
    parser.add_argument("--model", type=str, default="results/best_model.pth", help="Path to trained model weights")
    
    args = parser.parse_args()
    
    classifier = AcanthosisClassifier(model_path=args.model)
    res = classifier.predict(args.image, bbox=args.bbox)
    
    print("\n--- Inference Result ---")
    print(f"Image       : {res['image_path']}")
    print(f"Prediction  : {res['prediction']}")
    print(f"Confidence  : {res['confidence']:.2f}%")
    print(f"AN Prob     : {res['probability_an']:.4f}")
