import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as vision_models
import onnxruntime as ort
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
boneloss_src_path = os.path.join(BASE_DIR, "modules", "bone_loss")
if boneloss_src_path not in sys.path:
    sys.path.insert(0, boneloss_src_path)

from models.detector import ToothDetector
from models.landmark import PerioLandmarkPredictor
from utils.geometry import calculate_rbl
from utils.calibration import CalibrationManager
from .base_predictor import BasePanoramicPredictor

class BoneLossPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, detector_path: str, classifier_path: str, classifier_type: str, device: str, pixels_per_mm: float = 10.0):
        self.device = device
        self.classifier_type = classifier_type
        self.calibrator = CalibrationManager(pixels_per_mm=pixels_per_mm)
        self.detector = None
        self.landmark = None
        self.classifier = None
        self.load_model(detector_path, classifier_path)

    def load_model(self, detector_path: str, classifier_path: str) -> None:
        self.detector = ToothDetector(weights_path=detector_path, device=self.device)
        self.landmark = PerioLandmarkPredictor(device=self.device)
        
        if self.classifier_type == "onnx":
            self.classifier = ort.InferenceSession(classifier_path, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'])
        else:
            self.classifier = vision_models.mobilenet_v3_small()
            self.classifier.classifier[3] = torch.nn.Linear(self.classifier.classifier[3].in_features, 2)
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
            self.classifier = self.classifier.to(self.device)
            self.classifier.eval()

    def update_pixels_per_mm(self, pixels_per_mm: float):
        self.calibrator = CalibrationManager(pixels_per_mm=pixels_per_mm)

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        # image should be RGB numpy array
        pil_image = Image.fromarray(image)
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        cls_input = t(pil_image).unsqueeze(0)
        
        if self.classifier_type == "pytorch":
            with torch.no_grad():
                _, preds = torch.max(self.classifier(cls_input.to(self.device)), 1)
                is_pano = (preds.item() == 1)
        else:
            ort_outs = self.classifier.run(None, {self.classifier.get_inputs()[0].name: cls_input.numpy()})
            is_pano = (np.argmax(ort_outs[0], axis=1)[0] == 1)

        if not is_pano:
            raise ValueError("OOD 필터: 파노라마 이미지가 아닙니다.")

        dets = self.detector.predict(image)
        metrics = []
        landmarks_data = []

        for d in dets:
            t_num, b = d["tooth_number"], d["bbox"]
            lms = self.landmark.predict_landmarks(image, b)
            m_rbl = calculate_rbl(lms["mesial_cej"], lms["mesial_crest"], lms["root_apex"])
            d_rbl = calculate_rbl(lms["distal_cej"], lms["distal_crest"], lms["root_apex"])
            max_rbl = max(m_rbl, d_rbl)
            
            m_mm = self.calibrator.pixel_to_mm(np.linalg.norm(np.array(lms["mesial_cej"]) - np.array(lms["mesial_crest"])))
            d_mm = self.calibrator.pixel_to_mm(np.linalg.norm(np.array(lms["distal_cej"]) - np.array(lms["distal_crest"])))
            max_mm = max(m_mm, d_mm)
            
            metrics.append({
                "tooth_number": t_num,
                "rbl_percent": round(max_rbl, 1),
                "loss_mm": round(max_mm, 2)
            })

            landmarks_data.append({
                "tooth_number": t_num,
                "bbox": b,
                "mesial_cej": lms["mesial_cej"],
                "distal_cej": lms["distal_cej"],
                "mesial_crest": lms["mesial_crest"],
                "distal_crest": lms["distal_crest"],
                "root_apex": lms["root_apex"]
            })

        return {
            "module_name": "bone_loss_measurement",
            "metrics": metrics,
            "landmarks": landmarks_data
        }
