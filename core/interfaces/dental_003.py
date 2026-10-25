import os
import sys
import cv2
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
module_path = os.path.join(repo_root, "modules", "Dental_003")
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from models.detector import ToothDetector
from models.landmark import PerioLandmarkPredictor
from services.staging import determine_patient_stage, determine_site_stage
from utils.calibration import CalibrationManager
from utils.geometry import calculate_rbl


class BoneLossEngine:
    """
    Dental_003 치조골 소실(Radiographic Bone Loss, RBL) 계측 및 2017 World Workshop 병기 판별 엔진.
    - 1차: Dental_008 치아 인스턴스 분할 윤곽선(Contour) + 방사선 밀도 프로파일(CEJ/Crest) 정밀 계측
    - 폴백: Dental_003 자체 YOLOv11-OBB 치아 검출기 + SAM(vit_b) 마스크 분할 연동
    """

    def __init__(self, detector_path: str = None, sam_path: str = None, device: str = None, pixels_per_mm: float = 10.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_path = detector_path
        self.sam_path = sam_path
        self.calibrator = CalibrationManager(pixels_per_mm=pixels_per_mm)
        self.detector = None
        if detector_path and os.path.exists(detector_path):
            try:
                self.detector = ToothDetector(weights_path=detector_path, device=self.device)
            except Exception:
                self.detector = None
        # Lazy SAM initialization so contour-based fast path does not block startup VRAM
        self.landmark_predictor = PerioLandmarkPredictor(
            device=self.device,
            checkpoint_path=sam_path,
            lazy_sam=True,
        )

    def to(self, device):
        self.device = str(device)
        return self


def init_003_model():
    """Dental_003 (치조골 소실 계측 엔진)을 초기화합니다."""
    models_dir = os.path.join(module_path, "models")
    weights_dir = os.path.join(module_path, "weights")

    det_candidates = [
        os.path.join(models_dir, "best.pt"),
        os.path.join(models_dir, "best.onnx"),
        os.path.join(weights_dir, "best.pt"),
        os.path.join(weights_dir, "best.onnx"),
    ]
    sam_candidates = [
        os.path.join(models_dir, "sam_vit_b_01ec64.pth"),
        os.path.join(models_dir, "sam_vit_b_01ec64.onnx"),
        os.path.join(weights_dir, "sam_vit_b_01ec64.onnx"),
    ]
    det_path = next((p for p in det_candidates if os.path.exists(p)), None)
    sam_path = next((p for p in sam_candidates if os.path.exists(p)), None)

    return BoneLossEngine(detector_path=det_path, sam_path=sam_path)


def calculate_bone_loss(image: np.ndarray, tooth_roi_data: dict, model) -> dict:
    """
    영구치(11~48) 치아별 CEJ-Crest-Apex 방사선학적 골소실(RBL %) 및 물리적 흡수량(mm)을 계산하고,
    임상적으로 유의미한 골소실(Stage II 이상: RBL >= 15.0% 및 흡수량 >= 2.0mm) 부위의
    해부학적 치조골 소실 폴리곤([mesial_cej, distal_cej, distal_crest, mesial_crest])을 반환합니다.
    """
    if model is None:
        return {"contours": [], "metrics": [], "status": "UNIMPLEMENTED_MODEL"}

    if image.ndim == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()

    contours_out = []
    metrics_out = []
    landmarks_out = []
    tooth_stage_inputs = []

    fdi_labels = tooth_roi_data.get("fdi_labels", []) if isinstance(tooth_roi_data, dict) else []
    boxes = tooth_roi_data.get("boxes", []) if isinstance(tooth_roi_data, dict) else []
    contours = tooth_roi_data.get("contours", []) if isinstance(tooth_roi_data, dict) else []
    super_flags = tooth_roi_data.get("is_supernumerary", [False] * len(boxes)) if isinstance(tooth_roi_data, dict) else []

    h_img, w_img = image_rgb.shape[:2]
    all_teeth_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for cnt in contours:
        if cnt is not None and len(cnt) >= 3:
            pts_cnt = np.asarray(cnt, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(all_teeth_mask, [pts_cnt], 255)

    # Panorama width-adaptive pixel-to-mm calibration (nominal 10 px/mm at 2000px width)
    eff_px_per_mm = float(model.calibrator.pixels_per_mm) * max(0.8, float(w_img) / 2000.0)

    if fdi_labels and boxes:
        for idx, (fdi, box) in enumerate(zip(fdi_labels, boxes)):
            fdi_int = int(fdi)
            is_sup = bool(super_flags[idx]) if idx < len(super_flags) else (fdi_int in {19, 29, 39, 49})
            # 과잉치 및 제3대구치(사랑니 18, 28, 38, 48)는 치주염 RBL 정량 기준에서 제외
            if is_sup or fdi_int in {18, 28, 38, 48} or not (11 <= fdi_int <= 48):
                continue

            contour = contours[idx] if idx < len(contours) else None
            if contour is not None and len(contour) >= 6:
                lms = model.landmark_predictor.predict_landmarks_from_contour(
                    image_rgb, contour, box, tooth_number=fdi_int, external_teeth_mask=all_teeth_mask
                )
            else:
                x1, y1, x2, y2 = box
                cx_cy_wh = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]
                lms = model.landmark_predictor.predict_landmarks(
                    image_rgb, cx_cy_wh, tooth_number=fdi_int
                )

            try:
                m_rbl = calculate_rbl(lms["mesial_cej"], lms["mesial_crest"], lms["root_apex"])
                d_rbl = calculate_rbl(lms["distal_cej"], lms["distal_crest"], lms["root_apex"])
            except ValueError:
                continue

            max_rbl = max(m_rbl, d_rbl)
            m_px = float(np.linalg.norm(np.array(lms["mesial_cej"]) - np.array(lms["mesial_crest"])))
            d_px = float(np.linalg.norm(np.array(lms["distal_cej"]) - np.array(lms["distal_crest"])))
            m_mm = m_px / eff_px_per_mm
            d_mm = d_px / eff_px_per_mm
            max_mm = max(m_mm, d_mm)
            site_stage = determine_site_stage(max_rbl)
            conf = float(lms.get("confidence", 0.78))

            tooth_stage_inputs.append({"tooth": fdi_int, "max_rbl": max_rbl})
            metrics_out.append({
                "tooth_number": fdi_int,
                "mesial_rbl_percent": round(m_rbl, 1),
                "distal_rbl_percent": round(d_rbl, 1),
                "rbl_percent": round(max_rbl, 1),
                "loss_mm": round(max_mm, 2),
                "stage": site_stage.value,
                "confidence": round(conf, 2),
            })
            landmarks_out.append({
                "tooth_number": fdi_int,
                "mesial_cej": lms["mesial_cej"],
                "distal_cej": lms["distal_cej"],
                "mesial_crest": lms["mesial_crest"],
                "distal_crest": lms["distal_crest"],
                "root_apex": lms["root_apex"],
            })

            # 2017 World Workshop Stage II+ (RBL >= 15.0% and loss_mm >= 2.0mm) 부위를 폴리곤으로 표출
            if max_rbl >= 15.0 and max_mm >= 2.0:
                poly_pts = [
                    [float(lms["mesial_cej"][0]), float(lms["mesial_cej"][1])],
                    [float(lms["distal_cej"][0]), float(lms["distal_cej"][1])],
                    [float(lms["distal_crest"][0]), float(lms["distal_crest"][1])],
                    [float(lms["mesial_crest"][0]), float(lms["mesial_crest"][1])],
                ]
                contours_out.append({
                    "fdi": fdi_int,
                    "pts": poly_pts,
                    "rbl_percent": round(max_rbl, 1),
                    "loss_mm": round(max_mm, 2),
                    "stage": site_stage.value,
                    "confidence": round(conf, 2),
                })

    patient_stage, patient_extent = determine_patient_stage(tooth_stage_inputs)

    return {
        "contours": contours_out,
        "metrics": metrics_out,
        "landmarks": landmarks_out,
        "patient_stage": patient_stage.value,
        "patient_extent": patient_extent.value,
        "measurement_method": "radiographic_densitometric_cej_crest_v2",
        "status": "ACTIVE_INFERENCE",
    }
