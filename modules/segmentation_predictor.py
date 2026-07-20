import os
import numpy as np
import cv2
import torch

from ultralytics import YOLO
from modules.Dental_008.src.numbering.fdi_corrector import correct_fdi_numbers
from .base_predictor import BasePanoramicPredictor

class SegmentationPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
            else:
                print(f"Warning: Model not found at {self.model_path}. Segmentation predictor will not work.")
                
    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    def extract_contour(self, mask_tensor):
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        self.load_model()
        
        """
        Input: RGB or BGR numpy image
        Output: Dictionaries of detected teeth
        """
        # YOLOv8 handles RGB/BGR interchangeably but BGR is standard for cv2
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image
        
        conf = kwargs.get("conf", 0.5)
        iou = kwargs.get("iou", 0.4)
        
        if self.model is None:
            return {"module_name": "Dental_008_segmentation", "error": "Model not loaded"}

        results = self.model(img_bgr, verbose=False, conf=conf, iou=iou)
        res = results[0]
        
        if res.boxes is None or res.masks is None or len(res.boxes) == 0:
            return {
                "module_name": "Dental_008_segmentation",
                "teeth": []
            }
            
        boxes = res.boxes.xyxy.cpu().numpy()
        yolo_labels = res.boxes.cls.cpu().numpy()
        masks = res.masks.data
        
        # 기하학적 FDI 번호 보정 (DP 알고리즘)
        corrected_labels = correct_fdi_numbers(boxes, yolo_labels)
        
        teeth_data = []
        for i in range(len(boxes)):
            box = boxes[i]
            fdi = int(corrected_labels[i])
            if fdi <= 0: continue # 유효하지 않은 치식 무시
            
            contour = self.extract_contour(masks[i])
            
            teeth_data.append({
                "fdi": fdi,
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "contour": contour.tolist() if contour is not None else None
            })
            
        return {
            "module_name": "Dental_008_segmentation",
            "teeth": teeth_data
        }
