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
        self.restoration_classes = {
            1: 'Crown', 2: 'Filling', 3: 'Implant', 9: 'Root Canal Treatment',
            18: 'abutment', 25: 'post - core', 26: 'plating', 27: 'wire'
        }
        self.class_names = ['Crown', 'Filling', 'Implant', 'RCT', 'Other']

    def load_model(self) -> None:
        if self.yolo_model is None and self.session is None:
            if os.path.exists(self.model_path):
                # First attempt YOLO load (for best_restoration_model.onnx)
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

        teeth_data = kwargs.get("teeth_data", [])
        if not teeth_data:
            return {"module_name": "Dental_013_restoration", "results": []}

        restoration_results = []
        for tooth in teeth_data:
            fdi = tooth.get("fdi")
            bbox = tooth.get("bbox")
            if fdi is None or bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            tooth_crop = image[y1:y2, x1:x2]
            if tooth_crop.size == 0 or tooth_crop.shape[0] < 5 or tooth_crop.shape[1] < 5:
                continue

            if self.yolo_model is not None:
                try:
                    res = self.yolo_model.predict(source=tooth_crop, imgsz=640, conf=0.15, verbose=False)[0]
                    if len(res.boxes) > 0:
                        for cls_id_float, conf in zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                            cls_id = int(cls_id_float)
                            if cls_id in self.restoration_classes:
                                label = self.restoration_classes[cls_id]
                                restoration_results.append({
                                    "fdi": fdi,
                                    "restoration_type": label,
                                    "confidence": round(float(conf), 2)
                                })
                                break
                except Exception as e:
                    print(f"Dental_013 tooth {fdi} YOLO inference error: {e}")
            elif self.session is not None:
                if tooth_crop.shape[-1] == 3:
                    img_rgb = cv2.cvtColor(tooth_crop, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = tooth_crop
                img_resized = cv2.resize(img_rgb, (224, 224))
                img_scaled = img_resized.astype(np.float32) / 255.0
                img_chw = np.transpose(img_scaled, (2, 0, 1))
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                img_normalized = (img_chw - mean) / std
                input_tensor = np.expand_dims(img_normalized, axis=0)

                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: input_tensor})[0]
                exp_scores = np.exp(outputs[0] - np.max(outputs[0]))
                probabilities = exp_scores / np.sum(exp_scores)
                pred_idx = np.argmax(probabilities)
                pred_conf = float(probabilities[pred_idx])
                if pred_conf > 0.5:
                    pred_label = self.class_names[pred_idx] if pred_idx < len(self.class_names) else "Other"
                    restoration_results.append({
                        "fdi": fdi,
                        "restoration_type": pred_label,
                        "confidence": round(pred_conf, 2)
                    })

        return {
            "module_name": "Dental_013_restoration",
            "results": restoration_results
        }
