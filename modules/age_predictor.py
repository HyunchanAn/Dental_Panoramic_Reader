import os
import cv2
import numpy as np
import onnxruntime as ort

from .base_predictor import BasePanoramicPredictor

class AgePredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None

    def load_model(self) -> None:
        if self.session is None:
            if os.path.exists(self.model_path):
                # Load ONNX model with CPU and CUDA providers
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                try:
                    self.session = ort.InferenceSession(self.model_path, providers=providers)
                except Exception as e:
                    print(f"Error loading Age Predictor ONNX: {e}")
            else:
                print(f"Warning: Model not found at {self.model_path}. Age predictor will not work.")

    def unload_model(self) -> None:
        if self.session is not None:
            del self.session
            self.session = None

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        self.load_model()
        
        """
        Input: RGB or BGR numpy image
        Output: Dictionary containing predicted dental age
        """
        if self.session is None:
            return {"module_name": "Dental_011_age", "error": "Model not loaded"}

        teeth_data = kwargs.get("teeth_data", None)
        
        # If teeth_data is provided, mask the background
        if teeth_data is not None:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for t in teeth_data:
                contour = t.get("contour")
                if contour is not None and len(contour) > 0:
                    cv2.fillPoly(mask, [np.array(contour, dtype=np.int32)], 255)
            # Apply mask
            masked_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            masked_image = image

        # OpenCV BGR to RGB
        if masked_image.shape[-1] == 3:
            img_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = masked_image
            
        # Preprocess for ONNX (equivalent to torchvision transforms)
        # 1. Resize to (448, 224) (width, height)
        img_resized = cv2.resize(img_rgb, (448, 224))
        
        # 2. ToTensor: scale to [0, 1] and HWC -> CHW
        img_scaled = img_resized.astype(np.float32) / 255.0
        img_chw = np.transpose(img_scaled, (2, 0, 1))
        
        # 3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_normalized = (img_chw - mean) / std
        
        # 4. Add batch dimension: [1, 3, 224, 448]
        input_tensor = np.expand_dims(img_normalized, axis=0)
        
        # ONNX Inference
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})
        
        # output is a list of outputs. Assuming the first one is the age float
        pred_age = float(output[0][0][0]) if output[0].ndim > 1 else float(output[0][0])
            
        return {
            "module_name": "Dental_011_age",
            "predicted_age": round(pred_age, 2)
        }
