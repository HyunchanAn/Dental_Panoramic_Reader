import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .base_predictor import BasePanoramicPredictor

class PeriapicalPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        if self.model is None:
            if os.path.exists(self.model_path):
                # Load YOLO model (force task='detect' to avoid ONNX unpickling fallback errors)
                self.model = YOLO(self.model_path, task='detect')
            else:
                print(f"Warning: Model not found at {self.model_path}. Periapical predictor will not work.")

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        self.load_model()
        
        """
        Input: RGB or BGR numpy image
        Output: Dictionary containing detected periapical lesions and matched FDI
        """
        if self.model is None:
            return {"module_name": "Dental_012_periapical", "error": "Model not loaded"}

        # Run YOLO prediction
        results = self.model(image, conf=0.25, verbose=False)
        
        lesions = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0].item())
                
                lesion_data = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 2),
                    "fdi": None
                }
                lesions.append(lesion_data)
                
        # Match with FDI if teeth_data is provided
        teeth_data = kwargs.get("teeth_data", None)
        if teeth_data is not None and len(teeth_data) > 0 and len(lesions) > 0:
            self._match_fdi(lesions, teeth_data)
            
        return {
            "module_name": "Dental_012_periapical",
            "lesions": lesions
        }

    def _match_fdi(self, lesions: list, teeth_data: list):
        """
        치근단 병소(Periapical Lesion)를 해부학적 치근단(Apex) 앵커점을 기준으로 가장 인접한 치아에 정밀 매칭합니다.
        - 상악(11~28): 뿌리가 상방(Ymin)을 향하므로 치아 상단 치근단 앵커점과 매칭 및 하방 교차 페널티 적용
        - 하악(31~48): 뿌리가 하방(Ymax)을 향하므로 치아 하단 치근단 앵커점과 매칭 및 상방 교차 페널티 적용
        """
        for lesion in lesions:
            lx1, ly1, lx2, ly2 = lesion["bbox"]
            cx = (lx1 + lx2) / 2.0
            cy = (ly1 + ly2) / 2.0
            
            best_fdi = None
            min_dist = float('inf')
            
            for tooth in teeth_data:
                contour = tooth.get("contour")
                fdi = tooth.get("fdi")
                if fdi is None or contour is None or len(contour) == 0:
                    continue
                
                try:
                    fdi_num = int(fdi)
                except (ValueError, TypeError):
                    continue

                # 치아 BBox 범위 추출
                if hasattr(contour, 'reshape'):
                    pts = contour.reshape(-1, 2)
                    tx1, ty1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
                    tx2, ty2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
                else:
                    tx1 = float(min(p[0] for p in contour))
                    ty1 = float(min(p[1] for p in contour))
                    tx2 = float(max(p[0] for p in contour))
                    ty2 = float(max(p[1] for p in contour))
                
                tooth_cx = (tx1 + tx2) / 2.0
                tooth_w = max(10.0, tx2 - tx1)
                tooth_h = max(10.0, ty2 - ty1)

                is_maxillary = (11 <= fdi_num <= 28)
                is_mandibular = (31 <= fdi_num <= 48)

                if is_maxillary:
                    # 상악 치근단은 치아 상단 (ty1) 부근
                    apex_x = tooth_cx
                    apex_y = ty1
                    # 병소가 치관 하단보다 아래에 있으면 상악 병소 불가능
                    if cy > (ty1 + tooth_h * 0.7):
                        continue
                elif is_mandibular:
                    # 하악 치근단은 치아 하단 (ty2) 부근
                    apex_x = tooth_cx
                    apex_y = ty2
                    # 병소가 치관 상단보다 위에 있으면 하악 병소 불가능
                    if cy < (ty2 - tooth_h * 0.7):
                        continue
                else:
                    apex_x = tooth_cx
                    apex_y = (ty1 + ty2) / 2.0

                # 수평 오프셋(치아 축)과 수직 오프셋 계산
                dx = abs(cx - apex_x)
                dy = abs(cy - apex_y)

                # 치아 수평 폭의 1.5배 이상 벗어나면 매칭 제외
                if dx > (tooth_w * 1.5):
                    continue

                # 치근단 해부학적 가중 유클리디안 거리 (수평 정렬도에 더 높은 가중치)
                dist = np.sqrt((dx * 1.3) ** 2 + (dy * 0.9) ** 2)

                if dist < min_dist:
                    min_dist = dist
                    best_fdi = fdi_num
                        
            # 치아 크기 대비 허용 오차 내(치근단 부근 180px) 안착 시 FDI 할당
            if best_fdi is not None and min_dist < 180:
                lesion["fdi"] = best_fdi
