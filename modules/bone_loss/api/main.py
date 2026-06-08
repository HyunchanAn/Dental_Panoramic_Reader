"""
FastAPI Server Module for Pano_BoneLoss_Measurement.

방사선 이미지를 업로드받아 치아 검출 -> 랜드마크 추출 -> RBL 계산 -> 병기 판별
전체 파이프라인을 실행하고 결과를 JSON 형태로 반환합니다.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import sys
import os

# 내부 모듈 임포트 (패키지 절대/상대 경로 조정)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector import ToothDetector
from models.landmark import PerioLandmarkPredictor
from utils.geometry import calculate_rbl
from services.staging import determine_patient_stage, Stage, Extent

app = FastAPI(
    title="Pano_BoneLoss_Measurement API", 
    version="1.0.0", 
    description="Perio Staging & Bone Loss Auto-Measurement System API"
)

# 모델 초기화 (디바이스 설정: NVIDIA GeForce RTX 5080)
# 메모리 최적화를 위해 pinned memory 및 FP16 연산을 고려할 수 있습니다.
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = ToothDetector(weights_path="dummy_yolo.pt", device=device)
landmark_predictor = PerioLandmarkPredictor(backbone_type="swin").to(device)
landmark_predictor.eval()

@app.post("/predict/pano")
async def predict_panoramic(file: UploadFile = File(...)):
    """
    파노라마 방사선 이미지를 입력받아 치아별 골소실과 환자의 치주염 병기를 판별합니다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    # TODO: OpenCV/PIL을 이용한 이미지 바이트 로드 및 전처리 추가 필요
    dummy_tensor = torch.randn(1, 3, 1024, 2048).to(device)
    
    try:
        # 1. 치아 객체 탐지
        detections = detector.predict(dummy_tensor)
        
        tooth_metrics = []
        detailed_results = []
        
        for det in detections:
            tooth_num = det["tooth_number"]
            # 2. 크롭된 텐서 랜드마크 추출 (더미 입력)
            crop_tensor = torch.randn(1, 3, 224, 224).to(device)
            landmarks = landmark_predictor.predict_landmarks(crop_tensor)
            
            # 3. 골소실(RBL) 연산
            mesial_rbl = calculate_rbl(
                cej=landmarks["mesial_cej"], 
                crest=landmarks["mesial_crest"], 
                apex=landmarks["root_apex"]
            )
            distal_rbl = calculate_rbl(
                cej=landmarks["distal_cej"], 
                crest=landmarks["distal_crest"], 
                apex=landmarks["root_apex"]
            )
            
            max_rbl = max(mesial_rbl, distal_rbl)
            tooth_metrics.append({"tooth": tooth_num, "max_rbl": max_rbl})
            
            detailed_results.append({
                "tooth_number": tooth_num,
                "bbox": det["bbox"],
                "landmarks": landmarks,
                "mesial_rbl_percent": round(mesial_rbl, 2),
                "distal_rbl_percent": round(distal_rbl, 2),
                "max_rbl_percent": round(max_rbl, 2)
            })
            
        # 4. 전체 환자 레벨 병기 판별
        final_stage, extent = determine_patient_stage(
            tooth_metrics=tooth_metrics, 
            has_severe_complexity=False, 
            teeth_lost_due_to_perio=0
        )
        
        return JSONResponse(content={
            "status": "success",
            "global_diagnosis": {
                "stage": final_stage.value,
                "extent": extent.value
            },
            "tooth_level_results": detailed_results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

@app.post("/predict/periapical")
async def predict_periapical(file: UploadFile = File(...)):
    """
    치근단(Periapical) 방사선 이미지를 분석합니다.
    """
    return {"message": "Periapical analysis endpoint is initialized for future integration."}
