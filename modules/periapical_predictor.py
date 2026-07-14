import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .base_predictor import BasePanoramicPredictor

class PeriapicalPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        if os.path.exists(model_path):
            # Load YOLOv11 model
            self.model = YOLO(model_path)
        else:
            print(f"Warning: Model not found at {model_path}. Periapical predictor will not work.")

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        """
        Input: RGB or BGR numpy image
        Output: Dictionary containing detected periapical lesions and matched FDI
        """
        if self.model is None:
            return {"module_name": "Dental_012_periapical", "error": "Model not loaded"}

        # Run YOLO prediction
        results = self.model(image, conf=0.25, verbose=False)
        
        lesions = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                
                lesion_data = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 2),
                    "fdi": None
                }
                lesions.append(lesion_data)
                
        # Match with FDI if teeth_data is provided
        teeth_data = kwargs.get("teeth_data", None)
        if teeth_data is not None and len(teeth_data) > 0 and len(lesions) > 0:
            self._match_fdi(lesions, teeth_data)
            
        return {
            "module_name": "Dental_012_periapical",
            "lesions": lesions
        }

    def _match_fdi(self, lesions: list, teeth_data: list):
        """
        Match each lesion to the nearest tooth based on contour proximity.
        """
        for lesion in lesions:
            lx1, ly1, lx2, ly2 = lesion["bbox"]
            # Lesion center
            cx = (lx1 + lx2) / 2.0
            cy = (ly1 + ly2) / 2.0
            
            best_fdi = None
            min_dist = float('inf')
            
            for tooth in teeth_data:
                contour = tooth.get("contour")
                fdi = tooth.get("fdi")
                
                if contour is not None and len(contour) > 0:
                    # pointPolygonTest returns positive distance if inside, negative if outside
                    # We want to minimize the absolute distance if outside, or if inside, distance is 0 effectively
                    dist = cv2.pointPolygonTest(contour, (cx, cy), True)
                    
                    if dist >= 0:
                        # Inside the contour
                        actual_dist = 0
                    else:
                        # Outside the contour, dist is negative
                        actual_dist = abs(dist)
                        
                    if actual_dist < min_dist:
                        min_dist = actual_dist
                        best_fdi = fdi
                        
            # Assign the best FDI if it's reasonably close (e.g., within 200 pixels)
            if best_fdi is not None and min_dist < 200:
                lesion["fdi"] = best_fdi
