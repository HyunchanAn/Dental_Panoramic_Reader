import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as vision_models
import onnxruntime as ort
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
boneloss_src_path = os.path.join(BASE_DIR, "modules", "Dental_003")
if boneloss_src_path not in sys.path:
    sys.path.insert(0, boneloss_src_path)

from models.detector import ToothDetector
from models.landmark import PerioLandmarkPredictor
from utils.geometry import calculate_rbl
from utils.calibration import CalibrationManager
from .base_predictor import BasePanoramicPredictor

class BoneLossPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, detector_path: str, device: str, pixels_per_mm: float = 10.0):
        self.device = device
        self.calibrator = CalibrationManager(pixels_per_mm=pixels_per_mm)
        self.detector = None
        self.landmark = None
        self.load_model(detector_path)

    def load_model(self, detector_path: str) -> None:
        try:
            self.detector = ToothDetector(weights_path=detector_path, device=self.device)
            sam_path = os.path.join(boneloss_src_path, "models", "sam_vit_b_01ec64.pth")
            self.landmark = PerioLandmarkPredictor(device=self.device, checkpoint_path=sam_path)
        except FileNotFoundError as e:
            import logging
            logging.warning(f"Dental_003 weights not found: {e}. Wrapper initialized in degraded mode.")
            self.detector = None
            self.landmark = None

    def update_pixels_per_mm(self, pixels_per_mm: float):
        self.calibrator = CalibrationManager(pixels_per_mm=pixels_per_mm)

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        if self.detector is None or self.landmark is None:
            return {
                "module_name": "Dental_003_bone_loss_measurement",
                "status": "warning",
                "message": "BoneLoss models not loaded. Awaiting weights.",
                "metrics": [],
                "landmarks": []
            }
            
        teeth_data = kwargs.get("teeth_data", None)
        metrics = []
        landmarks_data = []

        if teeth_data is not None and len(teeth_data) > 0:
            # 008 모듈에서 제공한 치아 BBox와 FDI를 기준(SSOT)으로 사용
            for tooth in teeth_data:
                fdi = tooth.get("fdi")
                b = tooth.get("bbox")
                if fdi is None or b is None or len(b) != 4: continue
                
                lms = self.landmark.predict_landmarks(image, b)
                m_rbl = calculate_rbl(lms["mesial_cej"], lms["mesial_crest"], lms["root_apex"])
                d_rbl = calculate_rbl(lms["distal_cej"], lms["distal_crest"], lms["root_apex"])
                max_rbl = max(m_rbl, d_rbl)
                
                m_mm = self.calibrator.pixel_to_mm(np.linalg.norm(np.array(lms["mesial_cej"]) - np.array(lms["mesial_crest"])))
                d_mm = self.calibrator.pixel_to_mm(np.linalg.norm(np.array(lms["distal_cej"]) - np.array(lms["distal_crest"])))
                max_mm = max(m_mm, d_mm)
                
                metrics.append({"tooth_number": fdi, "rbl_percent": round(max_rbl, 1), "loss_mm": round(max_mm, 2)})
                landmarks_data.append({
                    "tooth_number": fdi, "bbox": b,
                    "mesial_cej": lms["mesial_cej"], "distal_cej": lms["distal_cej"],
                    "mesial_crest": lms["mesial_crest"], "distal_crest": lms["distal_crest"],
                    "root_apex": lms["root_apex"]
                })
        else:
            # Fallback: 독자 실행 시 내부 ToothDetector 사용
            dets = self.detector.predict(image)
            for d in dets:
                t_num, b = d["tooth_number"], d["bbox"]
                lms = self.landmark.predict_landmarks(image, b)
                m_rbl = calculate_rbl(lms["mesial_cej"], lms["mesial_crest"], lms["root_apex"])
                d_rbl = calculate_rbl(lms["distal_cej"], lms["distal_crest"], lms["root_apex"])
                max_rbl = max(m_rbl, d_rbl)
                
                m_mm = self.calibrator.pixel_to_mm(np.linalg.norm(np.array(lms["mesial_cej"]) - np.array(lms["mesial_crest"])))
                d_mm = self.calibrator.pixel_to_mm(np.linalg.norm(np.array(lms["distal_cej"]) - np.array(lms["distal_crest"])))
                max_mm = max(m_mm, d_mm)
                
                metrics.append({"tooth_number": t_num, "rbl_percent": round(max_rbl, 1), "loss_mm": round(max_mm, 2)})
                landmarks_data.append({
                    "tooth_number": t_num, "bbox": b,
                    "mesial_cej": lms["mesial_cej"], "distal_cej": lms["distal_cej"],
                    "mesial_crest": lms["mesial_crest"], "distal_crest": lms["distal_crest"],
                    "root_apex": lms["root_apex"]
                })

        return {
            "module_name": "Dental_003_bone_loss_measurement",
            "metrics": metrics,
            "landmarks": landmarks_data
        }
