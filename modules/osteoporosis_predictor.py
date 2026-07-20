import os
import numpy as np
import cv2
from PIL import Image

from modules.Dental_014.src.dental_014.inference import OsteoporosisInferencer
from .base_predictor import BasePanoramicPredictor

class OsteoporosisPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, weight_path: str):
        self.weight_path = weight_path
        self.inferencer = None

    def load_model(self) -> None:
        if self.inferencer is None:
            try:
                self.inferencer = OsteoporosisInferencer(weight_path=self.weight_path)
            except Exception as e:
                import logging
                logging.warning(f"Failed to load Osteoporosis End-to-End model: {e}")
                self.inferencer = None # 모델 학습 전이므로 None 허용

    def unload_model(self) -> None:
        if self.inferencer is not None:
            if hasattr(self.inferencer, 'model'):
                del self.inferencer.model
            del self.inferencer
            self.inferencer = None

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        self.load_model()
        
        if self.inferencer is None:
            return {
                "module_name": "Dental_014_osteoporosis",
                "status": "warning",
                "message": "Phase 2 Weights not loaded. Awaiting training completion.",
                "prediction": None
            }

        # Convert numpy RGB to PIL
        pil_img = Image.fromarray(image)
        
        pred_class, probs, mask, geom_feats = self.inferencer.predict(pil_img)

        return {
            "module_name": "Dental_014_osteoporosis",
            "status": "success",
            "prediction": {
                "class": pred_class,
                "probabilities": probs
            }
        }
