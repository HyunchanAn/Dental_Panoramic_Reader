import os
import cv2
import numpy as np
import onnxruntime as ort

from .base_predictor import BasePanoramicPredictor

class RestorationPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.class_names = ['Crown', 'Filling', 'Implant', 'RCT', 'Other']  # Based on Dental_013 classes

    def load_model(self) -> None:
        if self.session is None:
            if os.path.exists(self.model_path):
                try:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.session = ort.InferenceSession(self.model_path, providers=providers)
                except Exception as e:
                    print(f"Error loading Dental_013 ONNX model: {e}")
            else:
                print(f"Warning: Model not found at {self.model_path}. Restoration predictor will not work.")

    def unload_model(self) -> None:
        if self.session is not None:
            del self.session
            self.session = None

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        self.load_model()
        
        """
        Input: RGB or BGR numpy image (full panorama)
        Kwargs: teeth_data (from Dental_008) containing bounding boxes of each tooth.
        Output: Dictionary mapping FDI to Restoration classes.
        """
        if self.session is None:
            return {"module_name": "Dental_013_restoration", "error": "Model not loaded"}

        teeth_data = kwargs.get("teeth_data", [])
        if not teeth_data:
            return {"module_name": "Dental_013_restoration", "results": []}

        restoration_results = []
        
        for tooth in teeth_data:
            fdi = tooth.get("fdi")
            bbox = tooth.get("bbox") # [x1, y1, x2, y2]
            
            if fdi is None or bbox is None or len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure bbox is within image boundaries
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop tooth image
            tooth_crop = image[y1:y2, x1:x2]
            if tooth_crop.size == 0:
                continue
                
            if tooth_crop.shape[-1] == 3:
                # If BGR, convert to RGB
                img_rgb = cv2.cvtColor(tooth_crop, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = tooth_crop
                
            # Preprocess for ONNX (equivalent to torchvision transforms)
            # 1. Resize to (224, 224)
            img_resized = cv2.resize(img_rgb, (224, 224))
            
            # 2. ToTensor: scale to [0, 1] and HWC -> CHW
            img_scaled = img_resized.astype(np.float32) / 255.0
            img_chw = np.transpose(img_scaled, (2, 0, 1))
            
            # 3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            img_normalized = (img_chw - mean) / std
            
            # 4. Add batch dimension: [1, 3, 224, 224]
            input_tensor = np.expand_dims(img_normalized, axis=0)
            
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})[0]
            
            # Apply softmax
            exp_scores = np.exp(outputs[0] - np.max(outputs[0]))
            probabilities = exp_scores / np.sum(exp_scores)
            
            pred_idx = np.argmax(probabilities)
            pred_conf = float(probabilities[pred_idx])
                
            # If "Other" (e.g. natural tooth, post, etc.) is the highest, we might still include it,
            # but usually, we only report actual restorations with high confidence.
            if pred_conf > 0.5:
                pred_label = self.class_names[pred_idx]
                restoration_results.append({
                    "fdi": fdi,
                    "restoration_type": pred_label,
                    "confidence": round(pred_conf, 2)
                })

        return {
            "module_name": "Dental_013_restoration",
            "results": restoration_results
        }
