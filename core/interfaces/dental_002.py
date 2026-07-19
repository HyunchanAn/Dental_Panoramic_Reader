import os
import sys
import torch
from ultralytics import YOLO
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, "../../../Dental_002"))
if module_path not in sys.path:
    sys.path.append(module_path)

def init_002_model():
    """Dental_002 (Caries & Impacted) YOLOv11 모델을 초기화합니다."""
    model_path = os.path.join(module_path, "models", "best_refined.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(module_path, "models", "best.pt")
    if not os.path.exists(model_path):
        # fallback
        model_path = os.path.join(module_path, "runs", "detect", "train", "weights", "best.pt")
    
    # 모델 로드
    model = YOLO(model_path)
    return model

def run_caries_detection(image: np.ndarray, model) -> dict:
    """
    치아우식 및 매복치 병소를 탐지합니다.
    주의: 이 함수는 FDI 번호를 식별하지 않습니다. 병소의 좌표만 반환합니다.
    """
    results = model.predict(source=image, conf=0.25, iou=0.45, verbose=False)
    
    boxes = []
    labels = []
    scores = []
    
    if len(results) > 0:
        result = results[0]
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            boxes.append(box)
            labels.append(result.names[int(cls)])
            scores.append(float(conf))
            
    return {
        'boxes': boxes,
        'labels': labels, # 'Caries', 'Impacted' 등
        'scores': scores
    }
