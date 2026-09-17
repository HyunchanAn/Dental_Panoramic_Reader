import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, "../../../Dental_002"))
if module_path not in sys.path:
    sys.path.append(module_path)

def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """LAB L-channel CLAHE preprocessing for tooth patch contrast"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def init_002_model():
    """Dental_002 (Caries) YOLO 모델 초기화. 2-Stage 패치 모델 우선 탐색."""
    possible_paths = [
        os.path.join(current_dir, "../../modules/Dental_002/models/best_patch.onnx"),
        os.path.join(module_path, "models", "best_patch.onnx"),
        os.path.join(module_path, "models", "best_patch.pt"),
        os.path.join(module_path, "models", "best_refined.onnx"),
        os.path.join(module_path, "models", "best.onnx"),
        os.path.join(module_path, "models", "best.pt"),
    ]
    
    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        print("Warning: Dental_002 model weights not found. Running in fallback mode.")
        return None
        
    try:
        print(f"Loading Dental_002 model from: {model_path}")
        model = YOLO(model_path)
        model.is_patch_model = "patch" in os.path.basename(model_path).lower()
        return model
    except Exception as e:
        print(f"Warning: Failed to load Dental_002 YOLO model ({e}). Fallback mode active.")
        return None

def run_caries_detection(image: np.ndarray, model, tooth_boxes: list = None, margin: float = 0.15) -> dict:
    """
    치아우식(충치) 및 병소를 탐지합니다.
    tooth_boxes가 주어지면 2-Stage 고해상도 패치 기반 정밀 탐지 및 전역 좌표 역투영을 수행합니다.
    """
    boxes = []
    labels = []
    scores = []
    
    if model is None:
        return {'boxes': boxes, 'labels': labels, 'scores': scores}

    is_patch = getattr(model, "is_patch_model", False) or (tooth_boxes is not None and len(tooth_boxes) > 0)
    h, w = image.shape[:2]

    # [2-Stage Patch 기반 정밀 추론 및 파노라마 전역 역투영]
    if is_patch and tooth_boxes is not None and len(tooth_boxes) > 0:
        try:
            for tooth_box in tooth_boxes:
                tx1, ty1, tx2, ty2 = map(float, tooth_box[:4])
                tw = tx2 - tx1
                th = ty2 - ty1
                
                # +15% 마진 확장 (인접면 우식 보존)
                px1 = max(0, int(tx1 - tw * margin))
                py1 = max(0, int(ty1 - th * margin))
                px2 = min(w, int(tx2 + tw * margin))
                py2 = min(h, int(ty2 + th * margin))
                pw = px2 - px1
                ph = py2 - py1
                
                if pw <= 10 or ph <= 10:
                    continue
                    
                patch = image[py1:py2, px1:px2]
                patch_clahe = apply_clahe(patch)
                
                # 512x512 패치 정밀 추론
                results = model.predict(source=patch_clahe, imgsz=512, conf=0.20, iou=0.45, verbose=False)
                if len(results) > 0:
                    res = results[0]
                    for rbox, rcls, rconf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        rx1, ry1, rx2, ry2 = rbox
                        # 패치 내 512 상대 좌표 -> 원본 패치(pw, ph) 정규화 스케일링 및 전역 좌표 역투영
                        rx1_scaled = (rx1 / 512.0) * pw
                        ry1_scaled = (ry1 / 512.0) * ph
                        rx2_scaled = (rx2 / 512.0) * pw
                        ry2_scaled = (ry2 / 512.0) * ph

                        gx1 = float(max(px1, px1 + rx1_scaled))
                        gy1 = float(max(py1, py1 + ry1_scaled))
                        gx2 = float(min(px2, px1 + rx2_scaled))
                        gy2 = float(min(py2, py1 + ry2_scaled))

                        bw = gx2 - gx1
                        bh = gy2 - gy1
                        if bw <= 2 or bh <= 2:
                            continue
                        
                        boxes.append(np.array([gx1, gy1, gx2, gy2], dtype=np.float32))
                        labels.append(res.names[int(rcls)])
                        scores.append(float(rconf))
                        
            return {'boxes': boxes, 'labels': labels, 'scores': scores}
        except Exception as e:
            print(f"2-Stage Caries Patch detection error, falling back to 1-stage: {e}")

    # [1-Stage 전면 파노라마 안전 폴백: 단, 512 치아 패치 전용 모델인 경우 전체 파노라마 추론 금지]
    if not getattr(model, "is_patch_model", False):
        try:
            results = model.predict(source=image, conf=0.20, iou=0.40, verbose=False)
            if len(results) > 0:
                result = results[0]
                for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    boxes.append(box)
                    labels.append(result.names[int(cls)])
                    scores.append(float(conf))
        except Exception as e:
            print(f"Caries detection warning: {e}")
        
    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }