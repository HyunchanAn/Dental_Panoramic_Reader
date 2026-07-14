import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
missing_src_path = os.path.join(BASE_DIR, "modules", "Dental_010", "src")
if missing_src_path not in sys.path:
    sys.path.insert(0, missing_src_path)

from detector import MissingToothDetector
from .base_predictor import BasePanoramicPredictor

class MissingToothPredictorWrapper(BasePanoramicPredictor):
    def __init__(self):
        self.detector = None
        self.load_model()

    def load_model(self, *args, **kwargs) -> None:
        self.detector = MissingToothDetector()

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        """
        image 파라미터는 호환성을 위해 유지하지만, 실제 연산은 kwargs로 전달된 fdi_list를 기반으로 합니다.
        """
        fdi_list = kwargs.get("fdi_list", [])
        
        if not fdi_list:
            return {
                "module_name": "Dental_010_missing_tooth",
                "missing_teeth": [],
                "supernumerary_or_error": []
            }

        result = self.detector.detect(fdi_list)
        
        return {
            "module_name": "Dental_010_missing_tooth",
            "missing_teeth": result["missing_teeth"],
            "supernumerary_or_error": result["supernumerary_or_error"]
        }
