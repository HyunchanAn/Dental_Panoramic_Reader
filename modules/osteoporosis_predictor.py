import os
import sys
import numpy as np
import cv2
from PIL import Image

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _cand in [
    os.path.join(base_dir, "modules", "Dental_014", "src"),
    os.path.join(os.path.dirname(base_dir), "Dental_014", "src"),
]:
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
        break

from dental_014.inference import OsteoporosisInferencer
from .base_predictor import BasePanoramicPredictor


class OsteoporosisPredictorWrapper(BasePanoramicPredictor):
    def __init__(
        self, weight_path: str = None, model_path: str = None, device: str = None
    ):
        self.weight_path = weight_path or model_path
        self.device = device
        self.inferencer = None

    def load_model(self) -> None:
        if self.inferencer is None:
            if not self.weight_path or not os.path.exists(self.weight_path):
                return
            try:
                self.inferencer = OsteoporosisInferencer(
                    weight_path=self.weight_path, device=self.device
                )
            except Exception as e:
                import logging

                logging.warning(f"Failed to load Osteoporosis End-to-End model: {e}")
                self.inferencer = None

    def unload_model(self) -> None:
        if self.inferencer is not None:
            if hasattr(self.inferencer, "model"):
                del self.inferencer.model
            del self.inferencer
            self.inferencer = None

    def predict(self, image: np.ndarray, **kwargs) -> dict:
        self.load_model()

        if self.inferencer is None:
            return {
                "module_name": "Dental_014_osteoporosis",
                "status": "NOT_AVAILABLE",
                "score": None,
                "category": None,
                "message": "Osteoporosis weights not loaded.",
                "prediction": None,
            }

        try:
            pil_img = Image.fromarray(image)
            pred_class, probs, _, geom_feats = self.inferencer.predict(pil_img)

            p_normal = float(probs.get("Normal", 0.0))
            p_osteopenia = float(probs.get("Osteopenia (경도)", 0.0))
            p_osteoporosis = float(probs.get("Osteoporosis (중증)", 0.0))
            risk_score = round(float(p_osteopenia * 0.5 + p_osteoporosis * 1.0), 4)

            if pred_class == "Osteoporosis (중증)":
                category = "HIGH"
            elif pred_class == "Osteopenia (경도)":
                category = "MODERATE"
            else:
                category = "LOW"

            return {
                "module_name": "Dental_014_osteoporosis",
                "status": "ONLINE_INFERENCE",
                "score": risk_score,
                "category": category,
                "predicted_class": pred_class,
                "pred_class": pred_class,
                "probabilities": {
                    "Normal": round(p_normal, 4),
                    "Osteopenia": round(p_osteopenia, 4),
                    "Osteoporosis": round(p_osteoporosis, 4),
                },
                "morphology_features": geom_feats,
                "prediction": {
                    "class": pred_class,
                    "probabilities": probs,
                    "morphology": geom_feats,
                },
            }
        finally:
            self.unload_model()
