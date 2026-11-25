import os
import cv2
import numpy as np
import onnxruntime as ort

from .base_predictor import BasePanoramicPredictor

class RestorationPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.yolo_model = None
        self.session = None
        self.raw_class_names = {
            0: 'Caries', 1: 'Crown', 2: 'Filling', 3: 'Implant', 4: 'Malaligned',
            5: 'Mandibular Canal', 6: 'Missing teeth', 7: 'Periapical lesion',
            8: 'Retained root', 9: 'Root Canal Treatment', 10: 'Root Piece',
            11: 'impacted tooth', 12: 'maxillary sinus', 13: 'Bone Loss',
            14: 'Fracture teeth', 15: 'Permanent Teeth', 16: 'Supra Eruption',
            17: 'TAD', 18: 'abutment', 19: 'attrition', 20: 'bone defect',
            21: 'gingival former', 22: 'metal band', 23: 'orthodontic brackets',
            24: 'permanent retainer', 25: 'post - core', 26: 'plating', 27: 'wire',
            28: 'Cyst', 29: 'Root resorption', 30: 'Primary teeth'
        }
        self.crop_class_names = [
            "Bridge",
            "Crown",
            "Filling",
            "Implant",
            "Natural",
            "Other",
            "RCT",
        ]
        # SSOT 4-category clinical mapping
        self.category_mapping = {
            'Filling': 'Inlay/Onlay',
            'Bridge': 'Crown/Bridge',
            'Crown': 'Crown/Bridge',
            'metal band': 'Crown/Bridge',
            'plating': 'Crown/Bridge',
            'Implant': 'Implant',
            'TAD': 'Implant',
            'abutment': 'Implant',
            'gingival former': 'Implant',
            'RCT': 'Endodontic (RCT)',
            'Root Canal Treatment': 'Endodontic (RCT)',
            'post - core': 'Endodontic (RCT)',
            'orthodontic brackets': 'Other/Orthodontic',
            'permanent retainer': 'Other/Orthodontic',
            'wire': 'Other/Orthodontic'
        }

    def _infer_crop_onnx(self, image: np.ndarray, teeth_data: list, custom_mapping: dict) -> list:
        """Run 7-class EfficientNet-B0 224x224 ONNX classifier on individual tooth crops."""
        if self.session is None or not teeth_data:
            return []
        img_h, img_w = image.shape[:2]
        input_name = self.session.get_inputs()[0].name
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        out = []
        for tooth in teeth_data:
            fdi = tooth.get("fdi")
            bbox = tooth.get("bbox")
            if fdi is None or bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            tw, th = max(4.0, x2 - x1), max(4.0, y2 - y1)
            px1 = max(0, int(round(x1 - tw * 0.25)))
            py1 = max(0, int(round(y1 - th * 0.25)))
            px2 = min(img_w, int(round(x2 + tw * 0.25)))
            py2 = min(img_h, int(round(y2 + th * 0.25)))
            if px2 <= px1 or py2 <= py1:
                continue
            crop = image[py1:py2, px1:px2]
            if crop.ndim == 2:
                crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            else:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0
            chw = (np.transpose(crop, (2, 0, 1)) - mean) / std
            logits = self.session.run(None, {input_name: np.expand_dims(chw, axis=0)})[0][0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp)
            cls_idx = int(np.argmax(probs))
            conf = float(probs[cls_idx])
            raw_name = (
                self.crop_class_names[cls_idx]
                if cls_idx < len(self.crop_class_names)
                else f"Class_{cls_idx}"
            )
            clinical_cat = self.category_mapping.get(raw_name) or custom_mapping.get(raw_name)
            if clinical_cat is not None and conf >= 0.45:
                out.append({
                    "fdi": int(fdi) if str(fdi).isdigit() else fdi,
                    "raw_class": raw_name,
                    "clinical_category": clinical_cat,
                    "confidence": round(conf, 4),
                    "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                })
        return out

    def load_model(self) -> None:
        if self.yolo_model is None and self.session is None:
            if os.path.exists(self.model_path):
                try:
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(self.model_path, task='segment')
                    return
                except Exception as e:
                    print(f"YOLO load failed for {self.model_path}: {e}, falling back to ort.InferenceSession")

                try:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.session = ort.InferenceSession(self.model_path, providers=providers)
                except Exception as e:
                    print(f"Error loading Dental_013 ONNX model: {e}")
            else:
                print(f"Warning: Model not found at {self.model_path}. Restoration predictor will not work.")

    def unload_model(self) -> None:
        if self.yolo_model is not None:
            del self.yolo_model
            self.yolo_model = None
        if self.session is not None:
            del self.session
            self.session = None

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        self.load_model()
        if self.yolo_model is None and self.session is None:
            return {"module_name": "Dental_013_restoration", "error": "Model not loaded"}

        criteria = kwargs.get("criteria", {})
        custom_mapping = criteria.get("restoration_mapping", {})
        teeth_data = kwargs.get("teeth_data", [])

        # Step 1: Run ONNX crop classifier when self.session is active; otherwise YOLO-seg
        detected_restorations = []
        if self.session is not None and self.yolo_model is None:
            return {
                "module_name": "Dental_013_restoration",
                "results": self._infer_crop_onnx(image, teeth_data, custom_mapping),
            }
        if self.yolo_model is not None:
            try:
                res = self.yolo_model.predict(source=image, imgsz=640, conf=0.15, verbose=False)[0]
                if len(res.boxes) > 0:
                    for cls_id_float, conf, box in zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy()):
                        cls_id = int(cls_id_float)
                        raw_name = self.raw_class_names.get(cls_id, f"Class_{cls_id}")
                        
                        # Map to 4 SSOT clinical categories
                        clinical_cat = self.category_mapping.get(raw_name)
                        if clinical_cat is None and raw_name in custom_mapping:
                            clinical_cat = custom_mapping[raw_name]
                            
                        # Only record restoration / prosthetic classes
                        if clinical_cat is not None:
                            detected_restorations.append({
                                "raw_class": raw_name,
                                "clinical_category": clinical_cat,
                                "confidence": round(float(conf), 2),
                                "bbox": [int(x) for x in box]
                            })
            except Exception as e:
                print(f"Dental_013 YOLO full-image inference error: {e}")

        # Step 2: Spatial matching with tooth bounding boxes (FDI assignment)
        restoration_results = []
        for rest in detected_restorations:
            rx1, ry1, rx2, ry2 = rest["bbox"]
            rcx = (rx1 + rx2) / 2.0
            rcy = (ry1 + ry2) / 2.0
            r_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
            
            best_fdi = None
            max_inter = 0.0
            
            for tooth in teeth_data:
                fdi = tooth.get("fdi")
                bbox = tooth.get("bbox")
                if fdi is None or bbox is None or len(bbox) != 4:
                    continue
                tx1, ty1, tx2, ty2 = bbox
                
                # Check bounding box intersection
                ix1 = max(rx1, tx1)
                iy1 = max(ry1, ty1)
                ix2 = min(rx2, tx2)
                iy2 = min(ry2, ty2)
                
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area > max_inter:
                        max_inter = inter_area
                        best_fdi = int(fdi) if str(fdi).isdigit() else fdi
                elif tx1 <= rcx <= tx2 and ty1 <= rcy <= ty2:
                    # Center falls inside tooth
                    if max_inter == 0.0:
                        best_fdi = int(fdi) if str(fdi).isdigit() else fdi
            
            rest["fdi"] = best_fdi
            restoration_results.append(rest)

        return {
            "module_name": "Dental_013_restoration",
            "results": restoration_results
        }
