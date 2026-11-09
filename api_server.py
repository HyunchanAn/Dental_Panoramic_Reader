import os
import sys
import io
import time
import uuid
import hmac
import hashlib
import numpy as np
from PIL import Image, UnidentifiedImageError
from typing import Dict, Any, List, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DENTAL_CORE_DIR = os.path.join(BASE_DIR, "modules", "Dental_Core")
if os.path.isdir(DENTAL_CORE_DIR) and DENTAL_CORE_DIR not in sys.path:
    sys.path.insert(0, DENTAL_CORE_DIR)

try:
    from dental_core import strip_phi_tags as _core_strip_phi_tags
except ImportError:
    _core_strip_phi_tags = None

from core.pipeline import PanoramicPipeline
from modules.dental_010_heuristic import verify_missing_teeth

# Decompression bomb / memory exhaustion protection
Image.MAX_IMAGE_PIXELS = 50_000_000
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:8000,http://127.0.0.1:8000"
)


def get_allowed_origins() -> List[str]:
    raw_origins = os.environ.get(
        "DPR_CORS_ORIGINS",
        os.environ.get("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS),
    )
    origins = [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip() and origin.strip() != "*"
    ]
    return origins or ["http://localhost:5173", "http://127.0.0.1:5173"]


ALLOWED_ORIGINS = get_allowed_origins()

app = FastAPI(
    title="Dental Panoramic Reader API Gateway",
    description="MSA Backend Gateway API with Dynamic Midline & Gap Distance Ratio Correction",
    version="1.3.0",
)

# Restrict CORS to configured origins and disable credential sharing with wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Token"],
)

pipeline_instance: Optional[PanoramicPipeline] = None


def get_pipeline(use_004: bool = False) -> PanoramicPipeline:
    global pipeline_instance
    if pipeline_instance is None or pipeline_instance.use_004 != use_004:
        print(f"[API Gateway] Initializing PanoramicPipeline (use_004={use_004})...")
        pipeline_instance = PanoramicPipeline(use_004=use_004)
    return pipeline_instance


def strip_phi_tags_in_memory(ds: Any) -> Any:
    """Strip DICOM PHI tags in-memory prior to pixel extraction."""
    if _core_strip_phi_tags is not None:
        ds = _core_strip_phi_tags(ds)
    phi_tags = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "OtherPatientIDs",
        "PatientAddress",
        "PatientTelephoneNumbers",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "AccessionNumber",
        "StudyID",
    ]
    for tag in phi_tags:
        if hasattr(ds, tag):
            setattr(ds, tag, "ANONYMIZED")
    if hasattr(ds, "remove_private_tags"):
        try:
            ds.remove_private_tags()
        except Exception:
            pass
    return ds


def _is_dicom_payload(contents: bytes, filename: str = "", content_type: str = "") -> bool:
    lower_name = (filename or "").lower()
    lower_ct = (content_type or "").lower()
    if lower_name.endswith((".dcm", ".dicom")):
        return True
    if "dicom" in lower_ct:
        return True
    if len(contents) >= 132 and contents[128:132] == b"DICM":
        return True
    return False


def decode_uploaded_radiograph_in_memory(
    contents: bytes,
    filename: str = "",
    content_type: str = "",
) -> Tuple[np.ndarray, bool]:
    """
    Decode standard radiograph images (PNG/JPEG) or DICOM (.dcm) strictly in-memory
    via io.BytesIO without writing temporary files to disk.
    For DICOM inputs, PHI metadata tags are stripped in-memory before pixel decoding.
    Returns (rgb_uint8_array, is_dicom).
    """
    if _is_dicom_payload(contents, filename, content_type):
        try:
            import pydicom

            ds = pydicom.dcmread(io.BytesIO(contents), force=True)
            ds = strip_phi_tags_in_memory(ds)
            pixel_array = ds.pixel_array.astype(np.float32)
            p_min, p_max = float(np.min(pixel_array)), float(np.max(pixel_array))
            if p_max > p_min:
                norm_u8 = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                norm_u8 = np.zeros_like(pixel_array, dtype=np.uint8)
            if norm_u8.ndim == 2:
                rgb_arr = np.stack([norm_u8, norm_u8, norm_u8], axis=-1)
            elif norm_u8.ndim == 3 and norm_u8.shape[-1] == 1:
                rgb_arr = np.repeat(norm_u8, 3, axis=-1)
            elif norm_u8.ndim == 3 and norm_u8.shape[-1] >= 3:
                rgb_arr = norm_u8[:, :, :3]
            else:
                raise ValueError("Unsupported DICOM pixel array dimensions.")
            if rgb_arr.shape[0] * rgb_arr.shape[1] > Image.MAX_IMAGE_PIXELS:
                raise Image.DecompressionBombError("DICOM pixel dimensions exceed MAX_IMAGE_PIXELS.")
            return rgb_arr, True
        except Image.DecompressionBombError:
            raise
        except Exception as exc:
            raise UnidentifiedImageError(f"Invalid DICOM payload: {exc}") from exc

    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    return np.array(image_pil), False


def normalize_bbox(box: List[float], img_w: int, img_h: int) -> Dict[str, float]:
    """Convert absolute [x1, y1, x2, y2] to normalized {x, y, w, h} (0.0 ~ 1.0)"""
    x1, y1, x2, y2 = map(float, box[:4])
    return {
        "x": round(float(max(0.0, min(1.0, x1 / float(img_w)))), 4),
        "y": round(float(max(0.0, min(1.0, y1 / float(img_h)))), 4),
        "w": round(float(max(0.0, min(1.0, (x2 - x1) / float(img_w)))), 4),
        "h": round(float(max(0.0, min(1.0, (y2 - y1) / float(img_h)))), 4),
    }


def calculate_dynamic_midline(tooth_roi_data: Dict[str, Any], default_w: int) -> float:
    """
    Calculate dynamic midline (X_mid) based on detected anterior teeth (11, 21, 31, 41)
    to prevent head rotation / asymmetry drift errors.
    """
    if not tooth_roi_data or "fdi_labels" not in tooth_roi_data or "boxes" not in tooth_roi_data:
        return default_w / 2.0

    anterior_fdis = {11, 21, 31, 41}
    mid_xs = []

    for fdi, box in zip(tooth_roi_data["fdi_labels"], tooth_roi_data["boxes"]):
        if fdi in anterior_fdis:
            cx = (box[0] + box[2]) / 2.0
            mid_xs.append(cx)

    if len(mid_xs) > 0:
        return float(np.mean(mid_xs))
    return default_w / 2.0


def _build_module_execution_status(raw_report: Dict[str, Any], fdi_status: str) -> Dict[str, Dict[str, Optional[str]]]:
    raw_exec = raw_report.get("module_execution", {})
    status_map: Dict[str, Dict[str, Optional[str]]] = {}

    def _entry(mod_id: str, default_status: str, default_reason: Optional[str] = None) -> Dict[str, Optional[str]]:
        if isinstance(raw_exec, dict) and mod_id in raw_exec and isinstance(raw_exec[mod_id], dict):
            return {
                "status": str(raw_exec[mod_id].get("status", default_status)),
                "reason": raw_exec[mod_id].get("reason", default_reason),
            }
        return {"status": default_status, "reason": default_reason}

    has_008 = "008_tooth_data" in raw_report and raw_report["008_tooth_data"] is not None
    status_map["Dental_008"] = _entry(
        "Dental_008",
        "executed" if (has_008 and fdi_status != "FDI_UNAVAILABLE") else "unavailable",
        None if (has_008 and fdi_status != "FDI_UNAVAILABLE") else "FDI segmentation unavailable",
    )
    status_map["Dental_002"] = _entry(
        "Dental_002",
        "executed" if "002_lesions" in raw_report and raw_report["002_lesions"] is not None else "unavailable",
        None if "002_lesions" in raw_report and raw_report["002_lesions"] is not None else "Caries detector not executed",
    )

    if raw_report.get("has_deciduous") is True:
        status_map["Dental_003"] = _entry("Dental_003", "skipped", "Deciduous dentition detected")
    elif raw_report.get("003_bone_loss") is not None:
        status_map["Dental_003"] = _entry("Dental_003", "executed", None)
    else:
        status_map["Dental_003"] = _entry("Dental_003", "unavailable", "Bone loss module weights unavailable or not executed")

    status_map["Dental_009"] = _entry(
        "Dental_009",
        "executed" if "009_impacted" in raw_report and raw_report["009_impacted"] is not None else "skipped",
        None if "009_impacted" in raw_report and raw_report["009_impacted"] is not None else "Impaction analyzer not executed",
    )
    status_map["Dental_010"] = _entry(
        "Dental_010",
        "executed" if fdi_status == "FDI_ACTIVE" else "skipped",
        None if fdi_status == "FDI_ACTIVE" else "Skipped because FDI tooth coordinates are unavailable",
    )

    age_raw = raw_report.get("011_age")
    age_ok = isinstance(age_raw, dict) and age_raw.get("predicted_age") is not None
    status_map["Dental_011"] = _entry(
        "Dental_011",
        "executed" if age_ok else "unavailable",
        None if age_ok else (age_raw.get("error") if isinstance(age_raw, dict) else "Age estimation model unavailable"),
    )

    p_raw = raw_report.get("012_periapical")
    p_ok = p_raw is not None and not (isinstance(p_raw, dict) and p_raw.get("error"))
    status_map["Dental_012"] = _entry(
        "Dental_012",
        "executed" if p_ok else "unavailable",
        None if p_ok else "Periapical lesion model unavailable",
    )

    r_raw = raw_report.get("013_restoration")
    r_ok = r_raw is not None and not (isinstance(r_raw, dict) and r_raw.get("error"))
    status_map["Dental_013"] = _entry(
        "Dental_013",
        "executed" if r_ok else "unavailable",
        None if r_ok else "Restoration model unavailable",
    )

    osteo_raw = raw_report.get("014_osteoporosis")
    osteo_ok = isinstance(osteo_raw, dict) and "pred_class" in osteo_raw
    status_map["Dental_014"] = _entry(
        "Dental_014",
        "executed" if osteo_ok else "unavailable",
        None if osteo_ok else "Osteoporosis screening model unavailable",
    )

    return status_map


def format_to_ssot_report(
    raw_report: Dict[str, Any],
    img_w: int,
    img_h: int,
    filename: str,
    image_hash: str = "",
    preprocessing_id: str = "PRE-VISTA-AUTO-LETTERBOX-v1",
) -> Dict[str, Any]:
    caries_list = []
    bone_loss_list = []
    periapical_list = []

    # If super-resolution (004_image) was executed, all downstream coordinates are in 004_image pixel space
    sr_img = raw_report.get("004_image")
    if isinstance(sr_img, np.ndarray) and sr_img.ndim >= 2:
        pred_h, pred_w = int(sr_img.shape[0]), int(sr_img.shape[1])
    else:
        pred_w, pred_h = img_w, img_h

    tooth_roi = raw_report.get("008_tooth_data", {})
    midline_x_pred = calculate_dynamic_midline(tooth_roi, pred_w)
    midline_x = midline_x_pred * (float(img_w) / float(pred_w)) if pred_w > 0 else float(img_w) / 2.0

    # 1. Caries & Lesions mapping (Dental_002) with uncertain_fdi & relative position
    if "002_lesions" in raw_report and raw_report["002_lesions"]:
        for lesion in raw_report["002_lesions"]:
            box = lesion.get("box", [0, 0, 0, 0])
            norm_box = normalize_bbox(box, pred_w, pred_h)
            conf = round(float(lesion.get("confidence", 0.0)), 4)

            fdi = lesion.get("fdi", "Unknown")
            uncertain = conf < 0.65 or fdi == "Unknown" or fdi is None

            cx = (box[0] + box[2]) / 2.0
            quadrant = "Q1/Q4 (Right)" if cx < midline_x_pred else "Q2/Q3 (Left)"
            relative_label = f"{quadrant} RelPos-{(abs(cx - midline_x_pred) / (pred_w / 2.0)):.2f}"

            if uncertain:
                fdi_label = f"Quadrant Approx ({relative_label})"
                fdi_num = None
            else:
                try:
                    fdi_num = int(fdi)
                    fdi_label = f"FDI-{fdi_num}"
                except ValueError:
                    fdi_num = None
                    fdi_label = str(fdi)
                    uncertain = True

            caries_list.append(
                {
                    "x": norm_box["x"],
                    "y": norm_box["y"],
                    "w": norm_box["w"],
                    "h": norm_box["h"],
                    "confidence": conf,
                    "label": lesion.get("lesion_type", "Dental Caries"),
                    "toothNumber": fdi_num,
                    "fdi_label": fdi_label,
                    "relative_label": relative_label,
                    "uncertain_fdi": uncertain,
                }
            )

    # 2. Bone Loss mapping (Dental_003)
    if "003_bone_loss" in raw_report and raw_report["003_bone_loss"]:
        bl_data = raw_report["003_bone_loss"]
        if isinstance(bl_data, dict) and "contours" in bl_data:
            for contour_info in bl_data.get("contours", []):
                points = []
                for pt in contour_info.get("pts", []):
                    px, py = pt[0], pt[1]
                    points.append(
                        {
                            "x": round(max(0.0, min(1.0, px / pred_w)), 4),
                            "y": round(max(0.0, min(1.0, py / pred_h)), 4),
                        }
                    )

                bone_loss_list.append(
                    {
                        "points": points,
                        "confidence": contour_info.get("confidence", 0.88),
                        "label": "Alveolar Bone Loss",
                        "toothNumber": contour_info.get("fdi", None),
                        "rblPercent": contour_info.get("rbl_percent", None),
                        "lossMm": contour_info.get("loss_mm", None),
                        "stage": contour_info.get("stage", None),
                    }
                )

    # 3. Periapical Lesions (Dental_012)
    if "012_periapical" in raw_report and raw_report["012_periapical"]:
        p_raw = raw_report["012_periapical"]
        p_items = (
            p_raw.get("lesions", [])
            if isinstance(p_raw, dict)
            else (p_raw if isinstance(p_raw, list) else [])
        )
        for item in p_items:
            box = item.get("bbox", [0, 0, 0, 0])
            norm_box = normalize_bbox(box, pred_w, pred_h)
            periapical_list.append(
                {
                    "x": norm_box["x"],
                    "y": norm_box["y"],
                    "w": norm_box["w"],
                    "h": norm_box["h"],
                    "confidence": round(float(item.get("confidence", item.get("score", 0.80))), 4),
                    "label": "Periapical Lesion",
                    "toothNumber": item.get("fdi", None),
                    "clinicalStatus": item.get("clinical_status", "DEFINITE"),
                    "isolatedAnatomicalSuspicion": bool(item.get("isolated_anatomical_suspicion", False)),
                }
            )

    # 4. Detected Teeth (Dental_008) - 32 Tooth BBoxes, FDI Labels & Supernumerary Separation
    teeth_list = []
    supernumerary_list = []
    fdi_labels = tooth_roi.get("fdi_labels", []) if tooth_roi else []
    boxes = tooth_roi.get("boxes", []) if tooth_roi else []
    uncertain_flags = tooth_roi.get("uncertain", [False] * len(boxes)) if tooth_roi else []
    scores = tooth_roi.get("scores", [0.9] * len(boxes)) if tooth_roi else []
    super_flags = tooth_roi.get("is_supernumerary", [False] * len(boxes)) if tooth_roi else []

    for idx, (fdi, box, unc, sc) in enumerate(zip(fdi_labels, boxes, uncertain_flags, scores)):
        is_sup = bool(super_flags[idx]) if idx < len(super_flags) else (int(fdi) in {19, 29, 39, 49})
        norm_box = normalize_bbox(box, pred_w, pred_h)
        tooth_entry = {
            "toothNumber": int(fdi),
            "x": norm_box["x"],
            "y": norm_box["y"],
            "w": norm_box["w"],
            "h": norm_box["h"],
            "confidence": round(float(sc), 2),
            "uncertain": bool(unc),
            "isSupernumerary": is_sup,
        }
        teeth_list.append(tooth_entry)
        if is_sup:
            supernumerary_list.append(tooth_entry)

    # 5. Restorations & Prosthetics (Dental_013)
    restorations_list = []
    if "013_restoration" in raw_report and raw_report["013_restoration"]:
        r_raw = raw_report["013_restoration"]
        r_items = (
            r_raw.get("results", [])
            if isinstance(r_raw, dict)
            else (r_raw if isinstance(r_raw, list) else [])
        )
        for item in r_items:
            box = item.get("bbox", item.get("box", [0, 0, 0, 0]))
            norm_box = normalize_bbox(box, pred_w, pred_h)
            rest_type = item.get(
                "clinical_category",
                item.get("type", item.get("raw_class", item.get("class_name", "Restoration"))),
            )
            restorations_list.append(
                {
                    "toothNumber": item.get("fdi", None),
                    "type": rest_type,
                    "rawClass": item.get("raw_class", None),
                    "confidence": round(float(item.get("confidence", item.get("score", 0.85))), 2),
                    "x": norm_box["x"],
                    "y": norm_box["y"],
                    "w": norm_box["w"],
                    "h": norm_box["h"],
                }
            )

    # 6. Impacted Teeth (Dental_009)
    impacted_list = []
    if "009_impacted" in raw_report and raw_report["009_impacted"]:
        for item in raw_report["009_impacted"]:
            box = item.get("box", [0, 0, 0, 0])
            norm_box = normalize_bbox(box, pred_w, pred_h)
            impacted_list.append(
                {
                    "toothNumber": item.get("fdi"),
                    "wintersClass": item.get("winters_class", "Impacted"),
                    "eruptionStatus": item.get("eruption_status", "Fully Impacted"),
                    "angleDiff": item.get("angle_diff", 0.0),
                    "x": norm_box["x"],
                    "y": norm_box["y"],
                    "w": norm_box["w"],
                    "h": norm_box["h"],
                }
            )

    # 7. Missing Teeth Verification (Dental_010 Gap Ratio Heuristic)
    fdi_status = (
        tooth_roi.get("fdi_status", "FDI_ACTIVE")
        if isinstance(tooth_roi, dict) and tooth_roi
        else "FDI_UNAVAILABLE"
    )
    if fdi_status == "FDI_UNAVAILABLE":
        missing_analysis = {
            "verified_missing": [],
            "uncertain_missing": [],
            "gap_details": [],
            "status": "FDI_UNAVAILABLE",
        }
    else:
        missing_analysis = verify_missing_teeth(fdi_labels, boxes, pred_w, pred_h, midline_x_pred)
        missing_analysis["status"] = "FDI_ACTIVE"

    # 8. Clinical Synthesis & Odontogram Status Synthesis (32 teeth)
    all_32_teeth = [
        18, 17, 16, 15, 14, 13, 12, 11,
        21, 22, 23, 24, 25, 26, 27, 28,
        48, 47, 46, 45, 44, 43, 42, 41,
        31, 32, 33, 34, 35, 36, 37, 38,
    ]
    verified_missing_set = set(missing_analysis.get("verified_missing", []))
    periapical_fdis = {p.get("toothNumber") for p in periapical_list if p.get("toothNumber")}
    caries_fdi_map = {c.get("toothNumber"): c for c in caries_list if c.get("toothNumber")}
    bone_loss_fdis = {b.get("toothNumber") for b in bone_loss_list if b.get("toothNumber")}
    restoration_fdis = {r.get("toothNumber") for r in restorations_list if r.get("toothNumber")}
    impacted_fdis = {imp.get("toothNumber") for imp in impacted_list if imp.get("toothNumber")}

    odontogram = {}
    treatment_queue = []

    for fdi in all_32_teeth:
        if fdi in verified_missing_set:
            odontogram[str(fdi)] = {"status": "Missing", "label": "Missing"}
        elif fdi in periapical_fdis:
            odontogram[str(fdi)] = {"status": "Periapical", "label": "Periapical"}
            treatment_queue.append(
                {
                    "priority": "EMERGENT",
                    "fdi": fdi,
                    "condition": "Periapical Lesion",
                    "recommendation": f"FDI #{fdi} 치근단 병소: 정밀 방사선 및 근관 치료(Endodontics) 우선 고려 요망.",
                }
            )
        elif fdi in caries_fdi_map:
            conf = caries_fdi_map[fdi].get("confidence", 0.8)
            status_str = "Caries" if conf >= 0.45 else "Suspected"
            odontogram[str(fdi)] = {"status": status_str, "label": status_str}
            if conf >= 0.45:
                treatment_queue.append(
                    {
                        "priority": "HIGH",
                        "fdi": fdi,
                        "condition": "Dental Caries",
                        "recommendation": f"FDI #{fdi} 치아 우식증: 와동 형성 및 보철/수복(Restoration) 치료 권고.",
                    }
                )
        elif fdi in impacted_fdis:
            odontogram[str(fdi)] = {"status": "Impacted", "label": "Impacted"}
            treatment_queue.append(
                {
                    "priority": "HIGH",
                    "fdi": fdi,
                    "condition": "Impacted Tooth",
                    "recommendation": f"FDI #{fdi} 제3대구치 매복: 인접 치근 흡수 및 지치주위염 예방을 위한 외과적 발치 검토.",
                }
            )
        elif fdi in restoration_fdis:
            odontogram[str(fdi)] = {"status": "Restored", "label": "Restored"}
        elif fdi in bone_loss_fdis:
            odontogram[str(fdi)] = {"status": "BoneLoss", "label": "BoneLoss"}
            treatment_queue.append(
                {
                    "priority": "MODERATE",
                    "fdi": fdi,
                    "condition": "Alveolar Bone Loss",
                    "recommendation": f"FDI #{fdi} 치조골 흡수: 치주낭 계측 및 치근활택술/치주치료 권고.",
                }
            )
        else:
            odontogram[str(fdi)] = {"status": "Sound", "label": "Sound"}

    treatment_queue.append(
        {
            "priority": "PREVENTIVE",
            "fdi": None,
            "condition": "General Periodontal Care",
            "recommendation": "정기 치주 스케일링 및 6개월 단위 임상 파노라마 추적 관찰 추천.",
        }
    )

    clinical_synthesis = {
        "treatmentQueue": treatment_queue,
        "odontogram": odontogram,
    }

    # 8.5 Osteoporosis Risk Screening (Dental_014)
    osteo_raw = raw_report.get("014_osteoporosis")
    if osteo_raw and isinstance(osteo_raw, dict) and "pred_class" in osteo_raw:
        pred_cls = str(osteo_raw.get("pred_class", "Normal"))
        probs = osteo_raw.get("probabilities", {}) or {}
        if pred_cls.startswith("Osteoporosis"):
            osteo_category = "HIGH"
        elif pred_cls.startswith("Osteopenia"):
            osteo_category = "MODERATE"
        else:
            osteo_category = "LOW"
        normal_prob = float(probs.get("Normal", 0.85))
        osteo_score = round(max(0.0, min(1.0, 1.0 - normal_prob)), 4)
        osteo_risk = {
            "score": osteo_score,
            "category": osteo_category,
            "status": "ACTIVE_INFERENCE",
            "predictedClass": pred_cls,
            "probabilities": {k: round(float(v), 4) for k, v in probs.items()},
        }
    else:
        osteo_risk = {
            "score": None,
            "category": None,
            "status": "NOT_AVAILABLE",
            "predictedClass": None,
            "probabilities": None,
        }

    # 8.6 Dental Age Estimation (Dental_011)
    age_raw = raw_report.get("011_age")
    if isinstance(age_raw, dict) and age_raw.get("predicted_age") is not None:
        pred_age_val = round(float(age_raw["predicted_age"]), 2)
        age_estimation = {
            "predictedAge": pred_age_val,
            "confidenceInterval95": [
                round(max(0.0, pred_age_val - 1.38), 2),
                round(pred_age_val + 1.38, 2),
            ],
            "status": "ACTIVE_INFERENCE",
            "model": "ResNet50-AgeRegression-v1",
        }
    else:
        age_estimation = {
            "predictedAge": None,
            "confidenceInterval95": None,
            "status": "NOT_AVAILABLE",
            "model": None,
        }

    module_execution_status = _build_module_execution_status(raw_report, fdi_status)
    non_executed_modules = [
        f"{mod_id}({info['status']})"
        for mod_id, info in module_execution_status.items()
        if info.get("status") in ("skipped", "unavailable")
    ]

    summary_parts = []
    if teeth_list:
        summary_parts.append(f"치아 {len(teeth_list)}개 식별 완료.")
    if supernumerary_list:
        summary_parts.append(
            f"과잉치 {len(supernumerary_list)}개 ({[s['toothNumber'] for s in supernumerary_list]}) 분리 식별."
        )
    if impacted_list:
        summary_parts.append(
            f"매복치 {len(impacted_list)}개 ({[i['toothNumber'] for i in impacted_list]}) 감지."
        )
    if restorations_list:
        summary_parts.append(f"수복물/보철물 {len(restorations_list)}건 확인.")
    if caries_list:
        summary_parts.append(f"치아 우식증 {len(caries_list)}건 탐지.")
    if periapical_list:
        summary_parts.append(f"치근단 병소 {len(periapical_list)}건 탐지.")
    if bone_loss_list:
        summary_parts.append(f"치조골 소실 {len(bone_loss_list)}건 계측.")
    if missing_analysis["verified_missing"]:
        summary_parts.append(
            f"결손치(확정) {len(missing_analysis['verified_missing'])}개 ({missing_analysis['verified_missing']})."
        )
    if missing_analysis["uncertain_missing"]:
        summary_parts.append(f"결손치(유보/최후방) {len(missing_analysis['uncertain_missing'])}개 격리.")
    if age_estimation["predictedAge"] is not None:
        summary_parts.append(f"추정 치아 연령 약 {age_estimation['predictedAge']:.1f}세.")

    if summary_parts:
        summary_text = " ".join(summary_parts)
        if non_executed_modules:
            summary_text += f" [미실행/제한 모듈: {', '.join(non_executed_modules)}]"
    else:
        if non_executed_modules:
            summary_text = (
                "일부 진단 모듈이 미실행 또는 가중치 미로드 상태이므로 전체 정상 판정을 보류합니다 "
                f"({', '.join(non_executed_modules)})."
            )
        else:
            summary_text = "전 모듈 분석 완료: 특이 병소 소견이 관찰되지 않았습니다."

    anon_patient_id = (
        f"ANON-{image_hash[:12].upper()}"
        if image_hash
        else f"ANON-{uuid.uuid4().hex[:12].upper()}"
    )

    return {
        "reportId": f"REP-{int(time.time())}",
        "patientId": anon_patient_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "imageMetadata": {
            "filename": filename,
            "width": img_w,
            "height": img_h,
            "midline_x": round(midline_x, 1),
            "sha256_hash": image_hash,
            "preprocessing_id": preprocessing_id,
            "fdiStatus": fdi_status,
            "moduleVersions": raw_report.get("module_versions", {}),
            "moduleExecutionStatus": module_execution_status,
        },
        "findings": {
            "teeth": teeth_list,
            "supernumeraryTeeth": supernumerary_list,
            "caries": caries_list,
            "boneLoss": bone_loss_list,
            "periapicalLesions": periapical_list,
            "restorations": restorations_list,
            "impactedTeeth": impacted_list,
            "missingTeeth": missing_analysis,
            "osteoporosisRisk": osteo_risk,
            "ageEstimation": age_estimation,
            "moduleExecutionStatus": module_execution_status,
            "clinicalSynthesis": clinical_synthesis,
        },
        "summary": summary_text,
    }


@app.get("/api/v1/health")
def health_check():
    import torch

    gpu_available = torch.cuda.is_available()
    vram_free = 0.0
    if gpu_available:
        try:
            free_b, _ = torch.cuda.mem_get_info()
            vram_free = round(free_b / (1024 ** 3), 2)
        except Exception:
            pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipe_loaded = pipeline_instance is not None
    reg_models = pipeline_instance.manager.models if pipe_loaded and hasattr(pipeline_instance, "manager") else {}

    path_008_pt = os.path.join(base_dir, "modules", "Dental_008", "models", "yolov8m_best.pt")
    path_008_onnx = os.path.join(base_dir, "modules", "Dental_008", "models", "yolov8m_best.onnx")
    has_008 = os.path.exists(path_008_pt) or os.path.exists(path_008_onnx)
    weights_008 = (
        "yolov8m_best.pt"
        if os.path.exists(path_008_pt)
        else ("yolov8m_best.onnx" if os.path.exists(path_008_onnx) else "None")
    )

    path_002 = os.path.join(base_dir, "modules", "Dental_002", "models", "best_patch.onnx")
    has_002 = os.path.exists(path_002)

    path_003_onnx = os.path.join(base_dir, "modules", "Dental_003", "models", "best.onnx")
    path_003_pt = os.path.join(base_dir, "modules", "Dental_003", "models", "best.pt")
    path_003_sam = os.path.join(base_dir, "modules", "Dental_003", "models", "sam_vit_b_01ec64.pth")
    path_003_sam_onnx = os.path.join(base_dir, "modules", "Dental_003", "models", "sam_vit_b_01ec64.onnx")
    has_003 = (os.path.exists(path_003_onnx) or os.path.exists(path_003_pt)) and (
        os.path.exists(path_003_sam) or os.path.exists(path_003_sam_onnx)
    )
    weights_003 = "best.onnx + sam_vit_b" if has_003 else "None"

    path_011_onnx = os.path.join(base_dir, "modules", "Dental_011", "models", "age_resnet50.onnx")
    path_011_pth = os.path.join(base_dir, "modules", "Dental_011", "models", "best_model.pth")
    has_011 = os.path.exists(path_011_onnx) or os.path.exists(path_011_pth)
    weights_011 = (
        "age_resnet50.onnx"
        if os.path.exists(path_011_onnx)
        else ("best_model.pth" if os.path.exists(path_011_pth) else "None")
    )

    path_012_onnx = os.path.join(base_dir, "modules", "Dental_012", "models", "best.onnx")
    path_012_pt = os.path.join(base_dir, "modules", "Dental_012", "models", "best.pt")
    has_012 = os.path.exists(path_012_onnx) or os.path.exists(path_012_pt)
    weights_012 = (
        "best.pt"
        if os.path.exists(path_012_pt)
        else ("best.onnx" if os.path.exists(path_012_onnx) else "None")
    )

    path_013_pth = os.path.join(base_dir, "modules", "Dental_013", "models", "best_restoration_model.pth")
    path_013_onnx = os.path.join(base_dir, "modules", "Dental_013", "models", "best_restoration_model.onnx")
    has_013 = os.path.exists(path_013_onnx) or os.path.exists(path_013_pth)
    weights_013 = (
        "best_restoration_model.onnx"
        if os.path.exists(path_013_onnx)
        else ("best_restoration_model.pth" if os.path.exists(path_013_pth) else "Not Loaded")
    )

    path_014 = os.path.join(base_dir, "modules", "Dental_014", "models", "best.pt")
    has_014 = os.path.exists(path_014)

    has_009_code = os.path.exists(os.path.join(base_dir, "modules", "Dental_009", "src", "analyzer.py"))
    has_010_code = os.path.exists(os.path.join(base_dir, "modules", "dental_010_heuristic.py"))

    modules_status = [
        {
            "id": "Dental_008",
            "name": "치아 식별 및 FDI 분할",
            "type": "YOLOv8 Segmentation",
            "status": "ONLINE" if has_008 else "OFFLINE",
            "weights": weights_008,
            "version": "v1.2 (Tooth Cls)",
            "configured": True,
            "loaded": bool(pipe_loaded and "008" in reg_models),
            "ready": bool(has_008),
        },
        {
            "id": "Dental_002",
            "name": "치아 우식증 (2-Stage 패치)",
            "type": "YOLO Patch Detection",
            "status": "ONLINE" if has_002 else "OFFLINE",
            "weights": "best_patch.onnx" if has_002 else "None",
            "version": "v2.0 (Precision 83%)",
            "configured": True,
            "loaded": bool(pipe_loaded and "002" in reg_models),
            "ready": bool(has_002),
        },
        {
            "id": "Dental_012",
            "name": "치근단 병소 (음성증강)",
            "type": "YOLO11s Detection",
            "status": "ONLINE" if has_012 else "OFFLINE",
            "weights": weights_012,
            "version": "v2.0 (mAP 73.7%)",
            "configured": True,
            "loaded": bool(pipe_loaded and "012" in reg_models),
            "ready": bool(has_012),
        },
        {
            "id": "Dental_010",
            "name": "결손치 식별 및 갭 계측",
            "type": "Heuristic Gap Engine",
            "status": "ONLINE" if has_010_code else "OFFLINE",
            "weights": "Rule-based",
            "version": "v1.0 (Dynamic Midline)",
            "configured": True,
            "loaded": bool(has_010_code),
            "ready": bool(has_010_code and has_008),
        },
        {
            "id": "Dental_003",
            "name": "치조골 소실 계측",
            "type": "SAM + Radiodensity RBL",
            "status": "ONLINE" if has_003 else "STANDBY",
            "weights": weights_003,
            "version": "v2.0 (SAM + Densitometry)" if has_003 else "v1.0 (Standby)",
            "configured": True,
            "loaded": bool(pipe_loaded and "003" in reg_models),
            "ready": bool(has_003),
        },
        {
            "id": "Dental_009",
            "name": "매복 제3대구치 난이도 분석",
            "type": "Winter's Classification",
            "status": "ONLINE" if has_009_code else "OFFLINE",
            "weights": "Heuristic Geometry",
            "version": "v1.0 (Winter/Pell-Gregory)",
            "configured": True,
            "loaded": bool(has_009_code),
            "ready": bool(has_009_code and has_008),
        },
        {
            "id": "Dental_013",
            "name": "치과 수복물 분류",
            "type": "YOLOv8 Segmentation",
            "status": "ONLINE" if has_013 else "STANDBY",
            "weights": weights_013,
            "version": "v1.0 (YOLO ONNX)" if os.path.exists(path_013_onnx) else "v1.0",
            "configured": True,
            "loaded": bool(pipe_loaded and "013" in reg_models),
            "ready": bool(has_013),
        },
        {
            "id": "Dental_014",
            "name": "골다공증 위험도 스크리닝",
            "type": "ViT/MAE Mandibular Cortex Screening" if has_014 else "Mandibular Cortex Index (Unavailable)",
            "status": "ONLINE" if has_014 else "OFFLINE",
            "weights": "best.pt" if has_014 else "None",
            "version": "v1.0 (OsteoMAENet)" if has_014 else "v0.5 (Pending Weights)",
            "configured": True,
            "loaded": bool(pipe_loaded and "014" in reg_models),
            "ready": bool(has_014),
        },
        {
            "id": "Dental_011",
            "name": "치아 연령 추정",
            "type": "ResNet50 Age Regression",
            "status": "ONLINE" if has_011 else "STANDBY",
            "weights": weights_011,
            "version": "v1.0 (3-Way Split MAE 2.13y)",
            "configured": True,
            "loaded": bool(pipe_loaded and "011" in reg_models),
            "ready": bool(has_011),
        },
    ]

    if not has_008:
        overall_status = "UNHEALTHY"
    elif all(m["ready"] for m in modules_status):
        overall_status = "HEALTHY"
    else:
        overall_status = "DEGRADED"

    return {
        "status": overall_status,
        "service": "Dental_Panoramic_Reader API Gateway",
        "gpu_available": gpu_available,
        "vram_free_gb": vram_free,
        "pipeline_loaded": pipe_loaded,
        "modules": modules_status,
    }


def _verify_api_token(x_api_token: Optional[str], authorization: Optional[str]) -> None:
    """
    Fail-closed token verification:
    If DPR_API_TOKEN is unset or empty on the server, or if the client omits/mismatches
    the token, reject with HTTP 401 Unauthorized.
    """
    expected_token = os.environ.get("DPR_API_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: DPR_API_TOKEN is not configured on server (fail-closed).",
        )
    provided = (x_api_token or "").strip()
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if not provided or not hmac.compare_digest(
        provided.encode("utf-8"), expected_token.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Unauthorized: valid API token required.")


@app.post("/api/v1/infer")
def infer_panoramic(
    file: UploadFile = File(...),
    use_004: bool = Query(False, description="Enable Dental_004 Super-Resolution preprocessing"),
    x_api_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    _verify_api_token(x_api_token, authorization)

    filename = file.filename or "panoramic.png"
    content_type = file.content_type or ""
    is_dicom_hint = filename.lower().endswith((".dcm", ".dicom")) or "dicom" in content_type.lower()

    if content_type and not (
        content_type.startswith("image/")
        or is_dicom_hint
        or content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only image or DICOM (.dcm) files are supported.",
        )

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum allowed size ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    try:
        image_hash = hashlib.sha256(contents).hexdigest()
        img_np, is_dicom = decode_uploaded_radiograph_in_memory(
            contents, filename=filename, content_type=content_type
        )
        img_h, img_w = img_np.shape[:2]
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid or oversized image/DICOM file.")

    try:
        pipe = get_pipeline(use_004=use_004)
        raw_results = pipe.run(img_np)

        if is_dicom:
            preprocessing_id = (
                "PRE-DICOM-PHI-STRIPPED-SWINIR004-v1"
                if use_004
                else "PRE-DICOM-PHI-STRIPPED-LETTERBOX-v1"
            )
        else:
            preprocessing_id = (
                "PRE-VISTA-AUTO-LETTERBOX-SWINIR004-v1"
                if use_004
                else "PRE-VISTA-AUTO-LETTERBOX-v1"
            )

        final_report = format_to_ssot_report(
            raw_results,
            img_w,
            img_h,
            filename,
            image_hash=image_hash,
            preprocessing_id=preprocessing_id,
        )
        return final_report

    except HTTPException:
        raise
    except Exception:
        import traceback

        request_id = uuid.uuid4().hex[:12]
        print(f"[API Gateway ERROR] request_id={request_id}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal inference pipeline error (request_id={request_id}).",
        )


if __name__ == "__main__":
    import uvicorn

    allow_external = os.environ.get("ALLOW_EXTERNAL_ACCESS", "false").lower() in ("1", "true", "yes")
    bind_host = os.environ.get("API_HOST", "0.0.0.0" if allow_external else "127.0.0.1")
    bind_port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run("api_server:app", host=bind_host, port=bind_port, reload=False)
