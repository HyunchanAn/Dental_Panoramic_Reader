import os
import sys
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .base_predictor import BasePanoramicPredictor

class RestorationPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = ['Crown', 'Filling', 'Implant', 'RCT', 'Other']  # Based on Dental_013 classes
        self.load_model(model_path)
        
        # Transforms (must match the training transforms in Dental_013)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            try:
                # Assuming EfficientNet-B0 was used in Dental_013
                self.model = models.efficientnet_b0(weights=None)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_ftrs, len(self.class_names))
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Error loading Dental_013 model: {e}")
        else:
            print(f"Warning: Model not found at {model_path}. Restoration predictor will not work.")

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        """
        Input: RGB or BGR numpy image (full panorama)
        Kwargs: teeth_data (from Dental_008) containing bounding boxes of each tooth.
        Output: Dictionary mapping FDI to Restoration classes.
        """
        if self.model is None:
            return {"module_name": "Dental_013_restoration", "error": "Model not loaded"}

        teeth_data = kwargs.get("teeth_data", [])
        if not teeth_data:
            return {"module_name": "Dental_013_restoration", "results": []}

        restoration_results = []
        
        for tooth in teeth_data:
            fdi = tooth.get("fdi")
            bbox = tooth.get("bbox") # [x1, y1, x2, y2]
            
            if not fdi or not bbox:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure bbox is within image boundaries
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop tooth image
            tooth_crop = image[y1:y2, x1:x2]
            if tooth_crop.size == 0:
                continue
                
            # Convert to PIL Image for torchvision transforms
            if len(tooth_crop.shape) == 3 and tooth_crop.shape[2] == 3:
                # If BGR, convert to RGB
                # Assuming input to pipeline is already handled, let's make sure it's RGB
                # (pipeline.py usually passes current_img which is BGR if loaded by cv2)
                # But here we just convert BGR to RGB to be safe
                tooth_pil = Image.fromarray(tooth_crop[..., ::-1])
            else:
                tooth_pil = Image.fromarray(tooth_crop)
                
            input_tensor = self.transform(tooth_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_p, top_class = probabilities.topk(1, dim=1)
                
                pred_idx = top_class.item()
                pred_conf = top_p.item()
                
            # If "Other" (e.g. natural tooth, post, etc.) is the highest, we might still include it,
            # but usually, we only report actual restorations with high confidence.
            if pred_conf > 0.5:
                pred_label = self.class_names[pred_idx]
                restoration_results.append({
                    "fdi": fdi,
                    "restoration_type": pred_label,
                    "confidence": round(pred_conf, 2)
                })

        return {
            "module_name": "Dental_013_restoration",
            "results": restoration_results
        }
