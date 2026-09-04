import os
import io
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.pipeline import PanoramicPipeline
from modules.dental_010_heuristic import verify_missing_teeth

app = FastAPI(
    title="Dental Panoramic Reader API Gateway",
    description="MSA Backend Gateway API with Dynamic Midline & Gap Distance Ratio Correction",
    version="1.1.0",
)

# Enable CORS for Dental_015 frontend and local dev environments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_instance: Optional[PanoramicPipeline] = None

def get_pipeline(use_004: bool = False) -> PanoramicPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        print("[API Gateway] Initializing PanoramicPipeline...")
        pipeline_instance = PanoramicPipeline(use_004=use_004)
    return pipeline_instance


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
    if not tooth_roi_data or 'fdi_labels' not in tooth_roi_data or 'boxes' not in tooth_roi_data:
        return default_w / 2.0

    anterior_fdis = {11, 21, 31, 41}
    mid_xs = []
    
    for fdi, box in zip(tooth_roi_data['fdi_labels'], tooth_roi_data['boxes']):
        if fdi in anterior_fdis:
            cx = (box[0] + box[2]) / 2.0
            mid_xs.append(cx)

    if len(mid_xs) > 0:
        return float(np.mean(mid_xs))
    return default_w / 2.0


def format_to_ssot_report(raw_report: Dict[str, Any], img_w: int, img_h: int, filename: str) -> Dict[str, Any]:
    caries_list = []
    bone_loss_list = []
    periapical_list = []
    
    tooth_roi = raw_report.get('008_tooth_data', {})
    midline_x = calculate_dynamic_midline(tooth_roi, img_w)

    # 1. Caries & Lesions mapping (Dental_002) with uncertain_fdi & relative position
    if '002_lesions' in raw_report and raw_report['002_lesions']:
        for lesion in raw_report['002_lesions']:
            box = lesion.get('box', [0, 0, 0, 0])
            norm_box = normalize_bbox(box, img_w, img_h)
            conf = lesion.get('confidence', 0.85)
            
            fdi = lesion.get('fdi', 'Unknown')
            uncertain = conf < 0.65 or fdi == 'Unknown' or fdi is None
            
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            quadrant = "Q1/Q4 (Right)" if cx < midline_x else "Q2/Q3 (Left)"
            relative_label = f"{quadrant} RelPos-{(abs(cx - midline_x) / (img_w / 2.0)):.2f}"

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

            caries_list.append({
                "x": norm_box["x"],
                "y": norm_box["y"],
                "w": norm_box["w"],
                "h": norm_box["h"],
                "confidence": conf,
                "label": lesion.get('lesion_type', 'Dental Caries'),
                "toothNumber": fdi_num,
                "fdi_label": fdi_label,
                "relative_label": relative_label,
                "uncertain_fdi": uncertain,
            })

    # 2. Bone Loss mapping (Dental_003)
    if '003_bone_loss' in raw_report and raw_report['003_bone_loss']:
        bl_data = raw_report['003_bone_loss']
        if isinstance(bl_data, dict) and 'contours' in bl_data:
            for contour_info in bl_data.get('contours', []):
                points = []
                for pt in contour_info.get('pts', []):
                    px, py = pt[0], pt[1]
                    points.append({
                        "x": round(max(0.0, min(1.0, px / img_w)), 4),
                        "y": round(max(0.0, min(1.0, py / img_h)), 4),
                    })
                
                bone_loss_list.append({
                    "points": points,
                    "confidence": contour_info.get('confidence', 0.88),
                    "label": "Alveolar Bone Loss",
                    "toothNumber": contour_info.get('fdi', None),
                })

    # 3. Periapical Lesions (Dental_012)
    if '012_periapical' in raw_report and raw_report['012_periapical']:
        p_data = raw_report['012_periapical']
        if isinstance(p_data, list):
            for item in p_data:
                box = item.get('bbox', [0, 0, 0, 0])
                norm_box = normalize_bbox(box, img_w, img_h)
                periapical_list.append({
                    "x": norm_box["x"],
                    "y": norm_box["y"],
                    "w": norm_box["w"],
                    "h": norm_box["h"],
                    "confidence": item.get('score', 0.80),
                    "label": "Periapical Lesion",
                    "toothNumber": item.get('fdi', None),
                })

    # 4. Missing Teeth Verification (Dental_010 Gap Ratio Heuristic)
    fdi_labels = tooth_roi.get('fdi_labels', []) if tooth_roi else []
    boxes = tooth_roi.get('boxes', []) if tooth_roi else []
    missing_analysis = verify_missing_teeth(fdi_labels, boxes, img_w, img_h, midline_x)

    summary_parts = []
    if caries_list:
        summary_parts.append(f"치아 우식증 및 병소 {len(caries_list)}건 탐지.")
    if bone_loss_list:
        summary_parts.append(f"치조골 소실 부위 {len(bone_loss_list)}건 계측 완료.")
    if missing_analysis['verified_missing']:
        summary_parts.append(f"결손치(확정) {len(missing_analysis['verified_missing'])}개 식별 ({missing_analysis['verified_missing']}).")
    if missing_analysis['uncertain_missing']:
        summary_parts.append(f"결손치(유보/최후방) {len(missing_analysis['uncertain_missing'])}개 격리 처리.")

    summary_text = " ".join(summary_parts) if summary_parts else "분석 결과 특이 소견이 없습니다."

    return {
        "reportId": f"REP-{int(time.time())}",
        "patientId": "PATIENT-DEMO",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "imageMetadata": {
            "filename": filename,
            "width": img_w,
            "height": img_h,
            "midline_x": round(midline_x, 1),
        },
        "findings": {
            "caries": caries_list,
            "boneLoss": bone_loss_list,
            "periapicalLesions": periapical_list,
            "missingTeeth": missing_analysis,
            "osteoporosisRisk": {
                "score": 0.15,
                "category": "LOW",
            },
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
            free_b, total_b = torch.cuda.mem_get_info()
            vram_free = round(free_b / (1024 ** 3), 2)
        except Exception:
            pass
            
    # Check actual weights and module availability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    path_008 = os.path.join(base_dir, "modules", "Dental_008", "models", "yolov8m_best.pt")
    path_002 = os.path.join(base_dir, "modules", "Dental_002", "models", "best_patch.onnx")
    path_012 = os.path.join(base_dir, "modules", "Dental_012", "models", "best.onnx")
    path_013 = os.path.join(base_dir, "modules", "Dental_013", "models", "best_restoration_model.pth")
    
    modules_status = [
        {
            "id": "Dental_008",
            "name": "치아 식별 및 FDI 분할",
            "type": "YOLOv8 Segmentation",
            "status": "ONLINE" if os.path.exists(path_008) else "OFFLINE",
            "weights": "yolov8m_best.pt" if os.path.exists(path_008) else "None",
            "version": "v1.2 (Tooth Cls)",
        },
        {
            "id": "Dental_002",
            "name": "치아 우식증 (2-Stage 패치)",
            "type": "YOLO Patch Detection",
            "status": "ONLINE" if os.path.exists(path_002) else "OFFLINE",
            "weights": "best_patch.onnx" if os.path.exists(path_002) else "None",
            "version": "v2.0 (Precision 83%)",
        },
        {
            "id": "Dental_012",
            "name": "치근단 병소 (음성증강)",
            "type": "YOLO11s Detection",
            "status": "ONLINE" if os.path.exists(path_012) else "OFFLINE",
            "weights": "best.onnx" if os.path.exists(path_012) else "None",
            "version": "v2.0 (mAP 73.7%)",
        },
        {
            "id": "Dental_010",
            "name": "결손치 식별 및 갭 계측",
            "type": "Heuristic Gap Engine",
            "status": "ONLINE",
            "weights": "Rule-based",
            "version": "v1.0 (Dynamic Midline)",
        },
        {
            "id": "Dental_003",
            "name": "치조골 소실 계측",
            "type": "Bone Loss Masking",
            "status": "ONLINE",
            "weights": "Core Interface",
            "version": "v1.0",
        },
        {
            "id": "Dental_013",
            "name": "치과 수복물 분류",
            "type": "Restoration Classifier",
            "status": "STANDBY" if not os.path.exists(path_013) else "ONLINE",
            "weights": "best_restoration.pth" if os.path.exists(path_013) else "Not Loaded",
            "version": "v1.0",
        },
    ]

    return {
        "status": "HEALTHY",
        "service": "Dental_Panoramic_Reader API Gateway",
        "gpu_available": gpu_available,
        "vram_free_gb": vram_free,
        "pipeline_loaded": pipeline_instance is not None,
        "modules": modules_status,
    }


@app.post("/api/v1/infer")
def infer_panoramic(
    file: UploadFile = File(...),
    use_004: bool = Query(False, description="Enable Dental_004 Super-Resolution preprocessing"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are supported.")

    try:
        contents = file.file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image_pil)
        img_h, img_w = img_np.shape[:2]

        pipe = get_pipeline(use_004=use_004)
        raw_results = pipe.run(img_np)

        final_report = format_to_ssot_report(raw_results, img_w, img_h, file.filename or "panoramic.png")
        return final_report

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference Pipeline Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
