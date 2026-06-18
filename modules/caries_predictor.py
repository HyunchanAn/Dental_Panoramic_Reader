import sys
import os
import numpy as np
import cv2

# Submodule 경로 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caries_src_path = os.path.join(BASE_DIR, "modules", "caries_detection", "src")
if caries_src_path not in sys.path:
    sys.path.insert(0, caries_src_path)

from dentex_caries import CariesDetector
from .base_predictor import BasePanoramicPredictor

class CariesPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.detector = None
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        self.detector = CariesDetector(model_path=model_path)

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        # image expects RGB usually, but Caries predicts using BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image
        
        conf = kwargs.get("conf_c", 0.25)
        use_clahe = kwargs.get("use_clahe_c", True)
        clahe_clip = kwargs.get("clahe_clip_c", 2.0)
        use_sahi = kwargs.get("use_sahi_c", False)
        slice_size = kwargs.get("slice_size_c", 640)
        overlap_ratio = kwargs.get("overlap_ratio_c", 0.2)

        final_boxes, proc_bgr = self.detector.predict(
            img_bgr, use_clahe=use_clahe, clahe_clip=clahe_clip,
            use_sahi=use_sahi, slice_size=slice_size, overlap_ratio=overlap_ratio, conf=conf
        )

        predictions = []
        for item in final_boxes:
            predictions.append({
                "label": item["name"],
                "class_id": item["cls"],
                "confidence": item["conf"],
                "bbox": item["box"]  # [x1, y1, x2, y2]
            })

        return {
            "module_name": "caries_detection",
            "predictions": predictions,
            "processed_image_bgr": proc_bgr,
            "detector_ref": self.detector # For XAI explain later
        }
