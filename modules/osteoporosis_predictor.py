import sys
import os
import numpy as np
import cv2
from PIL import Image

# Submodule 경로 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
osteo_src_path = os.path.join(BASE_DIR, "modules", "Dental_014", "src")
if osteo_src_path not in sys.path:
    sys.path.insert(0, osteo_src_path)

from dental_014.inference import OsteoporosisInferencer
from .base_predictor import BasePanoramicPredictor

class OsteoporosisPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, weight_path: str):
        self.inferencer = None
        self.load_model(weight_path)

    def load_model(self, weight_path: str) -> None:
        try:
            self.inferencer = OsteoporosisInferencer(weight_path=weight_path)
        except FileNotFoundError:
            self.inferencer = None # 모델 학습 전이므로 None 허용

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        if self.inferencer is None:
            return {
                "module_name": "Dental_014_osteoporosis",
                "status": "warning",
                "message": "Phase 2 Weights not loaded. Awaiting training completion.",
                "prediction": None
            }

        # TODO: Phase 1 U-Net Crop Logic here
        # 현재는 입력 이미지를 직접 분류기로 전달 (테스트 목적)
        
        # Convert numpy RGB to PIL
        pil_img = Image.fromarray(image)
        
        pred_class, probs = self.inferencer.predict(pil_img)

        return {
            "module_name": "Dental_014_osteoporosis",
            "status": "success",
            "prediction": {
                "class": pred_class,
                "probabilities": probs
            }
        }
