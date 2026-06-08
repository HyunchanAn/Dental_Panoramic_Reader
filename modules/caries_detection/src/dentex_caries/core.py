import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from .preprocess import apply_clahe
from .explain import get_xai_heatmap

class CariesDetector:
    """
    치과 파노라마 이미지에서 충치 및 기타 병소를 탐지하는 통합 라이브러리 클래스.
    """
    def __init__(self, model_path="models/best_refined.pt", conf=0.25):
        self.model_path = model_path
        self.conf = conf
        self.model = YOLO(model_path)
        self.class_names = {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"}

    def predict(self, image, use_clahe=True, clahe_clip=2.0, use_sahi=False, slice_size=640, overlap_ratio=0.2, conf=None):
        """
        이미지에 대해 탐지를 수행합니다.
        
        Args:
            image (np.ndarray): BGR 이미지 (OpenCV format)
            use_clahe (bool): CLAHE 전처리 적용 여부
            use_sahi (bool): SAHI 슬라이싱 추론 사용 여부
            conf (float): Detection confidence threshold
        """
        processed_img = image.copy()
        if use_clahe:
            processed_img = apply_clahe(processed_img, clip_limit=clahe_clip)

        current_conf = conf if conf is not None else self.conf

        if use_sahi:
            # SAHI Inference
            sahi_model = UltralyticsDetectionModel(
                model_path=self.model_path,
                confidence_threshold=current_conf,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            result = get_sliced_prediction(
                processed_img,
                sahi_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio
            )
            # Convert SAHI results to a format compatible with our visualization
            preds = []
            for obj in result.object_prediction_list:
                bbox = obj.bbox.to_xyxy()
                preds.append({
                    "box": bbox,
                    "cls": obj.category.id,
                    "conf": obj.score.value,
                    "name": self.class_names.get(obj.category.id, "Unknown")
                })
            return preds, processed_img
        else:
            # Normal Inference
            results = self.model.predict(processed_img, conf=current_conf, verbose=False)
            preds = []
            for box in results[0].boxes:
                preds.append({
                    "box": box.xyxy[0].tolist(),
                    "cls": int(box.cls[0]),
                    "conf": float(box.conf[0]),
                    "name": self.class_names.get(int(box.cls[0]), "Unknown")
                })
            return preds, processed_img

    def explain(self, image, target_layer_idx=-4):
        """
        판단 근거를 위한 XAI 히트맵을 생성합니다.
        """
        return get_xai_heatmap(self.model_path, image, target_layer_idx)
