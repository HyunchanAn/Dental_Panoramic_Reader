import os
import numpy as np
import cv2

from modules.Dental_002.src.dentex_caries import CariesDetector
from .base_predictor import BasePanoramicPredictor

class CariesPredictorWrapper(BasePanoramicPredictor):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.detector = None

    def load_model(self) -> None:
        if self.detector is None:
            self.detector = CariesDetector(model_path=self.model_path)
            
    def unload_model(self) -> None:
        if self.detector is not None:
            # Force deletion of the InferenceSession if it exists
            if hasattr(self.detector, 'session'):
                del self.detector.session
            del self.detector
            self.detector = None

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        self.load_model()
        
        # image expects RGB usually, but Caries predicts using BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image
        
        conf = kwargs.get("conf_c", 0.25)
        use_clahe = kwargs.get("use_clahe_c", True)
        clahe_clip = kwargs.get("clahe_clip_c", 2.0)
        use_sahi = kwargs.get("use_sahi_c", False)
        slice_size = kwargs.get("slice_size_c", 640)
        overlap_ratio = kwargs.get("overlap_ratio_c", 0.2)

        final_boxes, proc_bgr = self.detector.predict(
            img_bgr, use_clahe=use_clahe, clahe_clip=clahe_clip,
            use_sahi=use_sahi, slice_size=slice_size, overlap_ratio=overlap_ratio, conf=conf
        )

        predictions = []
        for item in final_boxes:
            predictions.append({
                "label": item["name"],
                "class_id": item["cls"],
                "confidence": item["conf"],
                "bbox": item["box"],  # [x1, y1, x2, y2]
                "fdi": None
            })

        teeth_data = kwargs.get("teeth_data", None)
        if teeth_data is not None and len(teeth_data) > 0 and len(predictions) > 0:
            self._match_fdi(predictions, teeth_data)

        # Do not keep detector reference in the result dictionary to allow garbage collection
        return {
            "module_name": "Dental_002_caries_detection",
            "predictions": predictions,
            "processed_image_bgr": proc_bgr
        }
        
    def explain(self, image_path: str):
        self.load_model()
        viz, result = self.detector.explain(image_path)
        self.unload_model()
        return viz, result

    def _match_fdi(self, predictions: list, teeth_data: list):
        for item in predictions:
            x1, y1, x2, y2 = item["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            
            best_fdi = None
            min_dist = float('inf')
            
            for tooth in teeth_data:
                contour = tooth.get("contour")
                fdi = tooth.get("fdi")
                if contour is not None and len(contour) > 0:
                    dist = cv2.pointPolygonTest(np.array(contour, dtype=np.int32), (cx, cy), True)
                    if dist >= 0:
                        actual_dist = 0
                    else:
                        actual_dist = abs(dist)
                        
                    if actual_dist < min_dist:
                        min_dist = actual_dist
                        best_fdi = fdi
                        
            if best_fdi is not None and min_dist < 200:
                item["fdi"] = best_fdi
