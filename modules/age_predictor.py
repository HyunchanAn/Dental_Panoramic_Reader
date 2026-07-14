import sys
import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
age_src_path = os.path.join(BASE_DIR, "modules", "Dental_011", "src")
if age_src_path not in sys.path:
    sys.path.insert(0, age_src_path)

from model import AgeRegressionModel
from .base_predictor import BasePanoramicPredictor

class AgePredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            self.model = AgeRegressionModel(pretrained=False)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            print(f"Warning: Model not found at {model_path}. Age predictor will not work.")

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        """
        Input: RGB or BGR numpy image
        Output: Dictionary containing predicted dental age
        """
        if self.model is None:
            return {"module_name": "Dental_011_age", "error": "Model not loaded"}

        teeth_data = kwargs.get("teeth_data", None)
        
        # If teeth_data is provided, mask the background
        if teeth_data is not None:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for t in teeth_data:
                contour = t.get("contour")
                if contour is not None and len(contour) > 0:
                    cv2.fillPoly(mask, [contour], 255)
            # Apply mask
            masked_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            masked_image = image

        # OpenCV BGR to RGB
        if masked_image.shape[-1] == 3:
            img_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = masked_image
            
        pil_img = Image.fromarray(img_rgb)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_age = float(output.item())
            
        return {
            "module_name": "Dental_011_age",
            "predicted_age": round(pred_age, 2)
        }
