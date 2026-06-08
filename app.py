import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import torch
import gc
from huggingface_hub import hf_hub_download
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as vision_models
import torch.nn as nn
import onnxruntime as ort

st.set_page_config(page_title="Integrated AI Panoramic Reader", layout="wide")
st.title("Integrated AI Panoramic Radiograph Reader")
st.markdown("치아 우식 탐지와 치조골 소실 측정을 한 번에 제공하는 통합 분석 시스템입니다.")

st.sidebar.title("⚙️ Configuration")

with st.sidebar.expander("Caries Detection Settings", expanded=True):
    model_source_c = st.radio("Caries 모델", ["기본 모델", "사용자 학습 모델"])
    custom_model_c = st.text_input("모델 경로 (.pt)", "modules/caries_detection/models/best_refined.pt")
    model_path_c = custom_model_c if model_source_c != "기본 모델" else "modules/caries_detection/models/best_refined.pt"
    conf_threshold_c = st.slider("Confidence", 0.0, 1.0, 0.25)
    use_sahi_c = st.checkbox("Use SAHI", value=False)
    slice_size_c = st.select_slider("Slice Size", options=[320, 640, 800, 1024], value=640, disabled=not use_sahi_c)
    overlap_ratio_c = st.slider("Overlap Ratio", 0.0, 0.5, 0.2, 0.05, disabled=not use_sahi_c)
    use_clahe_c = st.checkbox("Apply CLAHE", value=True)
    clahe_clip_c = st.slider("CLAHE Clip", 0.0, 5.0, 2.0, 0.5, disabled=not use_clahe_c)
    show_xai_c = st.checkbox("Show XAI Heatmap", value=False)
    line_w_c = st.slider("Line Width", 1, 5, 2)

with st.sidebar.expander("Bone Loss Settings", expanded=True):
    pixels_per_mm = st.number_input("Pixels per mm (Calibration)", min_value=0.1, value=10.0, step=0.1)

import sys

# Submodule 경로를 시스템 패키지 경로로 추가하여 pip install 없이도 로드 가능하도록 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "modules", "caries_detection", "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "modules", "bone_loss"))

# Import submodules
try:
    from dentex_caries import CariesDetector
    from models.detector import ToothDetector
    from models.landmark import PerioLandmarkPredictor
    from utils.geometry import calculate_rbl
    from services.staging import determine_patient_stage
    from utils.calibration import CalibrationManager
except ImportError as e:
    st.error(f"모듈 로드 에러: 필수 의존성이 설치되었는지 확인하세요. {e}")
    st.stop()

@st.cache_resource
def load_caries_detector(path):
    if not os.path.exists(path):
        try: 
            path = hf_hub_download(repo_id="HyunchanAn/Caries_Detection_from_Panoramic", filename="best_refined.pt")
        except Exception as e:
            pass # fallback to local file
    return CariesDetector(model_path=path)

@st.cache_resource
def load_boneloss_models():
    device = "cpu"
    repo_id = "chemahc94/pano-boneloss-weights"
    
    def get_model_path(hf_name, local_path):
        if os.path.exists(local_path):
            return local_path
        try:
            return hf_hub_download(repo_id=repo_id, filename=hf_name)
        except Exception as e:
            # 원격에 파일이 없으면 조용히 로컬 폴백 경로를 사용함
            return local_path

    onnx_path = get_model_path("best.onnx", "modules/bone_loss/runs/detect/models/detector_train/weights/best.onnx")
    pt_path = get_model_path("best.pt", "modules/bone_loss/runs/detect/models/detector_train/weights/best.pt")
    
    final_weight = onnx_path if os.path.exists(onnx_path) else pt_path
    if not os.path.exists(final_weight):
        st.error(f"모델 파일이 존재하지 않습니다: {final_weight}. 네트워크를 확인하세요.")
        st.stop()
        
    detector = ToothDetector(weights_path=final_weight, device=device)
    landmark = PerioLandmarkPredictor(device=device)
    
    cls_onnx = get_model_path("pano_classifier.onnx", "modules/bone_loss/models/pano_classifier.onnx")
    if os.path.exists(cls_onnx):
        classifier = ort.InferenceSession(cls_onnx, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'])
        ctype = "onnx"
    else:
        cls_pt = get_model_path("pano_classifier.pt", "modules/bone_loss/models/pano_classifier.pt")
        if not os.path.exists(cls_pt):
            st.error(f"분류 모델 파일이 존재하지 않습니다: {cls_pt}")
            st.stop()
        classifier = vision_models.mobilenet_v3_small()
        classifier.classifier[3] = nn.Linear(classifier.classifier[3].in_features, 2)
        classifier.load_state_dict(torch.load(cls_pt, map_location=device))
        classifier = classifier.to(device)
        classifier.eval()
        ctype = "pytorch"
    return detector, landmark, classifier, ctype, device

caries_detector = load_caries_detector(model_path_c)
bone_detector, bone_landmark, bone_classifier, bone_ctype, bone_device = load_boneloss_models()
calibrator = CalibrationManager(pixels_per_mm=pixels_per_mm)

uploaded_file = st.file_uploader("이미지 업로드 (파노라마)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    st.image(image, caption='원본 이미지', use_container_width=True)
    tab1, tab2, tab3 = st.tabs(["우식 탐지 (Caries)", "치조골 소실 (Bone Loss)", "통합 리포트 (Integrated)"])
    
    with tab1:
        if st.button("우식 탐지 실행"):
            with st.spinner("Caries Detection 중..."):
                final_boxes, proc_bgr = caries_detector.predict(
                    img_bgr, use_clahe=use_clahe_c, clahe_clip=clahe_clip_c,
                    use_sahi=use_sahi_c, slice_size=slice_size_c, overlap_ratio=overlap_ratio_c, conf=conf_threshold_c
                )
                res_img = Image.fromarray(cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(res_img, "RGBA")
                colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 165, 0), 3: (128, 0, 128)}
                for item in final_boxes:
                    b, c, i = item["box"], item["conf"], item["cls"]
                    col = colors.get(i, (0, 0, 0))
                    draw.rectangle(b, outline=col, width=line_w_c)
                    draw.text((b[0], b[1] - 15), f"{item['name']} {c:.2f}", fill=col)
                st.image(res_img, use_container_width=True)
                if show_xai_c:
                    temp_img = "temp_xai.png"
                    image.save(temp_img)
                    viz, _ = caries_detector.explain(temp_img)
                    if viz is not None: st.image(viz, caption="XAI Heatmap")

    with tab2:
        if st.button("치조골 판독 실행"):
            with st.spinner("Bone Loss 측정 중..."):
                t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                cls_input = t(image).unsqueeze(0)
                if bone_ctype == "pytorch":
                    _, preds = torch.max(bone_classifier(cls_input.to(bone_device)), 1)
                    is_pano = (preds.item() == 1)
                else:
                    ort_outs = bone_classifier.run(None, {bone_classifier.get_inputs()[0].name: cls_input.numpy()})
                    is_pano = (np.argmax(ort_outs[0], axis=1)[0] == 1)
                
                if not is_pano:
                    st.error("OOD 필터: 파노라마 이미지가 아닙니다.")
                else:
                    dets = bone_detector.predict(img_rgb)
                    table_data = []
                    overlay = img_rgb.copy()
                    for d in dets:
                        t_num, b = d["tooth_number"], d["bbox"]
                        lms = bone_landmark.predict_landmarks(img_rgb, b)
                        m_rbl = calculate_rbl(lms["mesial_cej"], lms["mesial_crest"], lms["root_apex"])
                        d_rbl = calculate_rbl(lms["distal_cej"], lms["distal_crest"], lms["root_apex"])
                        max_rbl = max(m_rbl, d_rbl)
                        m_mm = calibrator.pixel_to_mm(np.linalg.norm(np.array(lms["mesial_cej"]) - np.array(lms["mesial_crest"])))
                        d_mm = calibrator.pixel_to_mm(np.linalg.norm(np.array(lms["distal_cej"]) - np.array(lms["distal_crest"])))
                        table_data.append({"Tooth": t_num, "RBL %": round(max_rbl, 1), "Loss (mm)": round(max(m_mm, d_mm), 2)})
                        
                        cx, cy, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        cv2.rectangle(overlay, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), (0, 255, 0), 2)
                        for pt, c in [(lms["mesial_cej"], (255,0,0)), (lms["distal_cej"], (255,0,0)), (lms["mesial_crest"], (0,255,255)), (lms["distal_crest"], (0,255,255)), (lms["root_apex"], (0,0,255))]:
                            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 5, c, -1)
                    
                    st.image(overlay, caption="Bone Loss Results")
                    st.dataframe(pd.DataFrame(table_data))

    with tab3:
        if st.button("통합 분석 실행 (Run All)"):
            st.info("두 모델을 모두 실행하여 한 화면에 결과를 합성합니다.")
            with st.spinner("통합 분석 중..."):
                final_boxes, proc_bgr = caries_detector.predict(img_bgr, conf=conf_threshold_c)
                overlay = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
                
                # Caries 결과 그리기
                colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 165, 0), 3: (128, 0, 128)}
                for item in final_boxes:
                    b, c, i = item["box"], item["conf"], item["cls"]
                    col = colors.get(i, (0, 0, 0))
                    cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, line_w_c)
                    cv2.putText(overlay, f"{item['name']} {c:.2f}", (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                dets = bone_detector.predict(img_rgb)
                
                upper_cej_pts, lower_cej_pts = [], []
                upper_crest_pts, lower_crest_pts = [], []
                mid_y = img_rgb.shape[0] // 2
                
                for d in dets:
                    lms = bone_landmark.predict_landmarks(img_rgb, d["bbox"])
                    cy = (d["bbox"][1] + d["bbox"][3]) / 2
                    
                    m_cej = (int(lms["mesial_cej"][0]), int(lms["mesial_cej"][1]))
                    d_cej = (int(lms["distal_cej"][0]), int(lms["distal_cej"][1]))
                    m_crest = (int(lms["mesial_crest"][0]), int(lms["mesial_crest"][1]))
                    d_crest = (int(lms["distal_crest"][0]), int(lms["distal_crest"][1]))
                    
                    tooth_num = d.get("tooth_number", 0)
                    if 11 <= tooth_num <= 28:  # 상악
                        upper_cej_pts.extend([m_cej, d_cej])
                        upper_crest_pts.extend([m_crest, d_crest])
                    elif 31 <= tooth_num <= 48:  # 하악
                        lower_cej_pts.extend([m_cej, d_cej])
                        lower_crest_pts.extend([m_crest, d_crest])
                    else:
                        # 예외 시 Y 좌표 폴백
                        if cy < mid_y:
                            upper_cej_pts.extend([m_cej, d_cej])
                            upper_crest_pts.extend([m_crest, d_crest])
                        else:
                            lower_cej_pts.extend([m_cej, d_cej])
                            lower_crest_pts.extend([m_crest, d_crest])
                        
                upper_cej_pts.sort(key=lambda p: p[0])
                lower_cej_pts.sort(key=lambda p: p[0])
                upper_crest_pts.sort(key=lambda p: p[0])
                lower_crest_pts.sort(key=lambda p: p[0])
                
                def draw_connected_line(pts, color):
                    for i in range(len(pts)-1):
                        cv2.line(overlay, pts[i], pts[i+1], color, 2)
                        
                draw_connected_line(upper_cej_pts, (255, 0, 0))     # CEJ 빨간선
                draw_connected_line(lower_cej_pts, (255, 0, 0))
                draw_connected_line(upper_crest_pts, (255, 165, 0)) # Crest 주황선
                draw_connected_line(lower_crest_pts, (255, 165, 0))
                        
                st.image(overlay, caption="통합 시각화 (병소 탐지 + 치조골 레벨)")
                st.success("통합 분석 완료!")
