"""
Streamlit Cloud Deployment UI for Pano_BoneLoss_Measurement.

이 모듈은 웹 브라우저 상에서 방사선 이미지를 업로드받고, 
내부 딥러닝 모듈과 기하학 연산 로직을 호출하여 결과를 시각화합니다.
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time
import torch.nn as nn
from torchvision import models, transforms

from models.detector import ToothDetector
from models.landmark import PerioLandmarkPredictor
from utils.geometry import calculate_rbl
from services.staging import determine_patient_stage, Stage, Extent
from utils.calibration import CalibrationManager
import onnxruntime as ort

# =====================================================================
# 페이지 설정 및 CSS
# =====================================================================
st.set_page_config(
    page_title="Pano BoneLoss Measurement",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 모델 초기화 (Streamlit Caching 활용)
# =====================================================================
@st.cache_resource
def load_models():
    # GPU 점유 문제를 우회하기 위해 기본적으로 CPU 모드를 강제합니다. (Issue 3 / GPU 미사용 요구조건)
    device = "cpu"
    
    # Hugging Face 설정 (본인의 repo_id로 변경 필요)
    repo_id = "chemahc94/pano-boneloss-weights"
    
    def download_if_missing(file_path, hf_filename):
        import os
        from huggingface_hub import hf_hub_download
        if not os.path.exists(file_path):
            print(f"{hf_filename} 이 없습니다. Hugging Face에서 다운로드합니다...")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            hf_hub_download(repo_id=repo_id, filename=hf_filename, local_dir=os.path.dirname(file_path))
            # local_dir을 쓰면 파일명 그대로 해당 폴더에 다운로드됨
    
    # 모델 경로 정의
    onnx_path = "runs/detect/models/detector_train/weights/best.onnx"
    pt_path = "runs/detect/models/detector_train/weights/best.pt"
    
    # 파일 확인 및 다운로드
    try:
        download_if_missing(onnx_path, "best.onnx")
    except Exception as e:
        print(f"ONNX 파일 다운로드 실패: {e}")
        
    try:
        download_if_missing(pt_path, "best.pt")
    except Exception as e:
        print(f"PT 파일 다운로드 실패: {e}")
    
    # YOLO ONNX 지원 (ONNX 파일이 있으면 사용, 없으면 pt 로드)
    import os
    if os.path.exists(onnx_path):
        detector = ToothDetector(weights_path=onnx_path, device=device)
    else:
        detector = ToothDetector(weights_path=pt_path, device=device)
        
    landmark_predictor = PerioLandmarkPredictor(device=device)
    
    # ONNX 런타임으로 MobileNetV3 로드 시도
    classifier_onnx_path = "models/pano_classifier.onnx"
    classifier_pt_path = "models/pano_classifier.pt"
    
    try:
        download_if_missing(classifier_onnx_path, "pano_classifier.onnx")
    except: pass
    try:
        download_if_missing(classifier_pt_path, "pano_classifier.pt")
    except: pass
    if os.path.exists(classifier_onnx_path):
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        classifier = ort.InferenceSession(classifier_onnx_path, providers=providers)
        classifier_type = "onnx"
    else:
        classifier = models.mobilenet_v3_small()
        num_ftrs = classifier.classifier[3].in_features
        classifier.classifier[3] = nn.Linear(num_ftrs, 2)
        classifier.load_state_dict(torch.load("models/pano_classifier.pt", map_location=device))
        classifier = classifier.to(device)
        classifier.eval()
        classifier_type = "pytorch"
    
    return detector, landmark_predictor, classifier, classifier_type, device

detector, landmark_predictor, classifier, classifier_type, device = load_models()

# =====================================================================
# UI 레이아웃
# =====================================================================
st.title("🦷 Pano BoneLoss Measurement System")
st.markdown("---")

# 사이드바: 컨트롤 패널
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "방사선 이미지 업로드 (Panoramic/Periapical)", 
        type=["png", "jpg", "jpeg", "dicom"]
    )
    
    st.markdown("---")
    st.header("Calibration Settings")
    st.info("실제 물리적 거리(mm) 산출을 위한 스케일 팩터를 입력하세요.")
    pixels_per_mm = st.number_input("Pixels per mm", min_value=0.1, value=10.0, step=0.1)
    calibrator = CalibrationManager(pixels_per_mm=pixels_per_mm)
    
    st.markdown("---")
    st.info("""
    **💡 딥러닝 추론 시스템 연동 완료**
    - 이미지 검증: 파노라마 사진 여부 판별 (ONNX Runtime 적용)
    - 치아 검출: YOLOv11 모델 적용. (ONNX Runtime)
    - 랜드마크 검출: SAM + 2-Stage ROI Crop (OOM 방지 및 CPU 최적화)
    """)

# 메인 화면: 결과 표출
if uploaded_file is not None:
    # 1. 이미지 로드 및 렌더링
    image_bytes = uploaded_file.read()
    # OpenCV 처리를 위해 numpy 배열로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    st.subheader("Original Radiograph")
    st.image(img_rgb, use_container_width=True)
    
    with st.spinner('파노라마 방사선 사진 여부 검증 및 분석 중...'):
        
        # 1.5 OOD Classification (입구 컷 필터)
        img_pil = Image.fromarray(img_rgb)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        cls_input = transform(img_pil).unsqueeze(0)
        if classifier_type == "pytorch":
            cls_input = cls_input.to(device)
            with torch.no_grad():
                outputs = classifier(cls_input)
                _, preds = torch.max(outputs, 1)
                is_pano = (preds.item() == 1) # 1이 'pano', 0이 'non_pano'
        else:
            # ONNX 추론
            input_name = classifier.get_inputs()[0].name
            ort_inputs = {input_name: cls_input.numpy()}
            ort_outs = classifier.run(None, ort_inputs)
            preds = np.argmax(ort_outs[0], axis=1)
            is_pano = (preds[0] == 1)
            
        if not is_pano:
            st.error("⚠️ 이 이미지는 파노라마 방사선 사진이 아닌 것 같습니다 (예: 치근단 방사선 사진, 일반 사진). 전체 치아 배열이 보이는 파노라마 원본을 업로드해 주세요.")
            st.stop()
            
        # 2. 모델 추론 파이프라인 (YOLOv11 적용)
        detections = detector.predict(img_rgb)
        
        tooth_metrics = []
        table_data = []
        
        # 시각화를 위한 복제 이미지
        overlay_img = img_rgb.copy()
        
        for det in detections:
            tooth_num = det["tooth_number"]
            bbox = det["bbox"] # [x_center, y_center, w, h, angle]
            
            # SAM 기반 랜드마크 추출 (원본 이미지 및 바운딩 박스 전달)
            landmarks = landmark_predictor.predict_landmarks(img_rgb, bbox)
            
            # RBL 연산 (%)
            mesial_rbl = calculate_rbl(landmarks["mesial_cej"], landmarks["mesial_crest"], landmarks["root_apex"])
            distal_rbl = calculate_rbl(landmarks["distal_cej"], landmarks["distal_crest"], landmarks["root_apex"])
            max_rbl = max(mesial_rbl, distal_rbl)
            
            # 절대적 거리 연산 (mm) - Issue 1
            mesial_loss_px = np.sqrt((landmarks["mesial_cej"][0] - landmarks["mesial_crest"][0])**2 + (landmarks["mesial_cej"][1] - landmarks["mesial_crest"][1])**2)
            distal_loss_px = np.sqrt((landmarks["distal_cej"][0] - landmarks["distal_crest"][0])**2 + (landmarks["distal_cej"][1] - landmarks["distal_crest"][1])**2)
            mesial_loss_mm = calibrator.pixel_to_mm(mesial_loss_px)
            distal_loss_mm = calibrator.pixel_to_mm(distal_loss_px)
            
            tooth_metrics.append({"tooth": tooth_num, "max_rbl": max_rbl})
            
            table_data.append({
                "Tooth (FDI)": tooth_num,
                "Mesial RBL (%)": round(mesial_rbl, 1),
                "Distal RBL (%)": round(distal_rbl, 1),
                "Max RBL (%)": round(max_rbl, 1),
                "Max Loss (mm)": round(max(mesial_loss_mm, distal_loss_mm), 2),
                "Status": "Normal" if max_rbl == 0 else ("Warning" if max_rbl < 33 else "Severe")
            })
            
            # [시각화] YOLO 바운딩 박스와 SAM 랜드마크 시각화
            cx, cy = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
            
            # 치아 번호 및 박스 (간이 표현)
            cv2.rectangle(overlay_img, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (0, 255, 0), 2)
            cv2.putText(overlay_img, f"#{tooth_num}", (cx - w//2, cy - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 랜드마크 점 찍기
            pts = [
                (landmarks["mesial_cej"], (255, 0, 0)),   # Blue (CEJ)
                (landmarks["distal_cej"], (255, 0, 0)),
                (landmarks["mesial_crest"], (0, 255, 255)), # Yellow (Crest)
                (landmarks["distal_crest"], (0, 255, 255)),
                (landmarks["root_apex"], (0, 0, 255))      # Red (Apex)
            ]
            for pt_coord, color in pts:
                pt_x, pt_y = int(pt_coord[0]), int(pt_coord[1])
                cv2.circle(overlay_img, (pt_x, pt_y), 5, color, -1)

        # 3. 환자 병기 판별
        final_stage, extent = determine_patient_stage(tooth_metrics)
        
    st.markdown("---")
    st.subheader("Analysis Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Global Periodontitis Stage", value=final_stage.value)
    with col2:
        st.metric(label="Disease Extent", value=extent.value)
        
    st.markdown("#### Tooth-level Metrics")
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("#### Detection Overlay")
    st.image(overlay_img, caption="Tooth Bounding Boxes & Landmarks (Dummy Visualization)", use_container_width=True)

else:
    st.info("왼쪽 사이드바에서 분석할 방사선 이미지를 업로드해 주세요.")
