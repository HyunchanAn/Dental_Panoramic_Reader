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
    custom_model_c = st.text_input("모델 경로 (.pt)", "modules/Dental_002/models/best_refined.pt")
    model_path_c = custom_model_c if model_source_c != "기본 모델" else "modules/Dental_002/models/best_refined.pt"
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
sys.path.insert(0, os.path.join(BASE_DIR, "modules", "Dental_002", "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "modules", "Dental_003"))

# Import registry and wrappers
try:
    from modules.registry import PredictorRegistry
    from modules.caries_predictor import CariesPredictorWrapper
    from modules.boneloss_predictor import BoneLossPredictorWrapper
except ImportError as e:
    st.error(f"모듈 로드 에러: {e}")
    st.stop()

@st.cache_resource
def load_registry(model_path_c):
    registry = PredictorRegistry()
    
    # Load Caries Wrapper
    c_path = model_path_c
    if not os.path.exists(c_path):
        try: 
            c_path = hf_hub_download(repo_id="chemahc94/Dental_002", filename="best_refined.pt")
        except Exception as e:
            pass # fallback
    caries_wrapper = CariesPredictorWrapper(model_path=c_path)
    registry.register_module("Dental_002_caries_detection", caries_wrapper)
    
    # Load Bone Loss Wrapper
    device = "cpu"
    repo_id = "chemahc94/Dental_003"  # Unified HF account
    def get_model_path(hf_name, local_path):
        if os.path.exists(local_path): return local_path
        try: return hf_hub_download(repo_id=repo_id, filename=hf_name)
        except Exception: return local_path

    onnx_path = get_model_path("best.onnx", "modules/Dental_003/runs/detect/models/detector_train/weights/best.onnx")
    pt_path = get_model_path("best.pt", "modules/Dental_003/runs/detect/models/detector_train/weights/best.pt")
    final_weight = onnx_path if os.path.exists(onnx_path) else pt_path
    
    cls_onnx = get_model_path("pano_classifier.onnx", "modules/Dental_003/models/pano_classifier.onnx")
    if os.path.exists(cls_onnx):
        cls_path, ctype = cls_onnx, "onnx"
    else:
        cls_path, ctype = get_model_path("pano_classifier.pt", "modules/Dental_003/models/pano_classifier.pt"), "pytorch"
        
    boneloss_wrapper = BoneLossPredictorWrapper(final_weight, cls_path, ctype, device)
    registry.register_module("Dental_003_bone_loss_measurement", boneloss_wrapper)
    
    return registry

registry = load_registry(model_path_c)

if 'pixels_per_mm' in locals():
    # Update calibration dynamically
    registry._predictors["Dental_003_bone_loss_measurement"].update_pixels_per_mm(pixels_per_mm)

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
                results = registry.run_pipeline(
                    img_bgr, ["Dental_002_caries_detection"], 
                    conf_c=conf_threshold_c, use_clahe_c=use_clahe_c, clahe_clip_c=clahe_clip_c, 
                    use_sahi_c=use_sahi_c, slice_size_c=slice_size_c, overlap_ratio_c=overlap_ratio_c
                )
                res = results.get("Dental_002_caries_detection", {})
                if res.get("status") == "error":
                    st.error(f"에러 발생: {res.get('error_message')}")
                else:
                    final_boxes = res.get("predictions", [])
                    proc_bgr = res.get("processed_image_bgr", img_bgr)
                    res_img = Image.fromarray(cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(res_img, "RGBA")
                    colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 165, 0), 3: (128, 0, 128)}
                    for item in final_boxes:
                        b, c, i, label = item["bbox"], item["confidence"], item["class_id"], item["label"]
                        col = colors.get(i, (0, 0, 0))
                        draw.rectangle(b, outline=col, width=line_w_c)
                        draw.text((b[0], b[1] - 15), f"{label} {c:.2f}", fill=col)
                    st.image(res_img, use_container_width=True)
                    if show_xai_c:
                        temp_img = "temp_xai.png"
                        image.save(temp_img)
                        viz, _ = res["detector_ref"].explain(temp_img)
                        if viz is not None: st.image(viz, caption="XAI Heatmap")

    with tab2:
        if st.button("치조골 판독 실행"):
            with st.spinner("Bone Loss 측정 중..."):
                results = registry.run_pipeline(img_rgb, ["Dental_003_bone_loss_measurement"])
                res = results.get("Dental_003_bone_loss_measurement", {})
                if res.get("status") == "error":
                    st.error(f"에러 발생: {res.get('error_message')}")
                else:
                    metrics = res.get("metrics", [])
                    landmarks = res.get("landmarks", [])
                    
                    table_data = []
                    overlay = img_rgb.copy()
                    
                    for lm_data in landmarks:
                        b = lm_data["bbox"]
                        cx, cy, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        cv2.rectangle(overlay, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), (0, 255, 0), 2)
                        
                        for pt, c in [(lm_data["mesial_cej"], (255,0,0)), (lm_data["distal_cej"], (255,0,0)), 
                                      (lm_data["mesial_crest"], (0,255,255)), (lm_data["distal_crest"], (0,255,255)), 
                                      (lm_data["root_apex"], (0,0,255))]:
                            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 5, c, -1)

                    for m in metrics:
                        table_data.append({"Tooth": m["tooth_number"], "RBL %": m["rbl_percent"], "Loss (mm)": m["loss_mm"]})
                    
                    st.image(overlay, caption="Bone Loss Results")
                    st.dataframe(pd.DataFrame(table_data))

    with tab3:
        if st.button("통합 분석 실행 (Run All)"):
            st.info("두 모델을 모두 실행하여 한 화면에 결과를 합성합니다.")
            with st.spinner("통합 분석 중..."):
                # run pipelines
                results_c = registry.run_pipeline(img_bgr, ["Dental_002_caries_detection"], conf_c=conf_threshold_c, use_clahe_c=use_clahe_c, clahe_clip_c=clahe_clip_c, use_sahi_c=use_sahi_c, slice_size_c=slice_size_c, overlap_ratio_c=overlap_ratio_c)
                results_b = registry.run_pipeline(img_rgb, ["Dental_003_bone_loss_measurement"])
                
                res_c = results_c.get("Dental_002_caries_detection", {})
                res_b = results_b.get("Dental_003_bone_loss_measurement", {})
                
                if res_c.get("status") == "error" or res_b.get("status") == "error":
                    if res_c.get("status") == "error": st.error(f"Caries Error: {res_c.get('error_message')}")
                    if res_b.get("status") == "error": st.error(f"BoneLoss Error: {res_b.get('error_message')}")
                    st.warning("일부 모듈에서 에러가 발생하여 정상 모듈만 표시될 수 있습니다.")
                
                overlay = img_rgb.copy()
                if res_c.get("status") == "success":
                    proc_bgr = res_c.get("processed_image_bgr", img_bgr)
                    overlay = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
                    colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 165, 0), 3: (128, 0, 128)}
                    for item in res_c.get("predictions", []):
                        b, c, i, label = item["bbox"], item["confidence"], item["class_id"], item["label"]
                        col = colors.get(i, (0, 0, 0))
                        cv2.rectangle(overlay, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, line_w_c)
                        cv2.putText(overlay, f"{label} {c:.2f}", (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                if res_b.get("status") == "success":
                    upper_cej_pts, lower_cej_pts = [], []
                    upper_crest_pts, lower_crest_pts = [], []
                    mid_y = img_rgb.shape[0] // 2
                    
                    for lm_data in res_b.get("landmarks", []):
                        b = lm_data["bbox"]
                        cy = (b[1] + b[3]) / 2
                        m_cej = (int(lm_data["mesial_cej"][0]), int(lm_data["mesial_cej"][1]))
                        d_cej = (int(lm_data["distal_cej"][0]), int(lm_data["distal_cej"][1]))
                        m_crest = (int(lm_data["mesial_crest"][0]), int(lm_data["mesial_crest"][1]))
                        d_crest = (int(lm_data["distal_crest"][0]), int(lm_data["distal_crest"][1]))
                        
                        tooth_num = lm_data.get("tooth_number", 0)
                        if 11 <= tooth_num <= 28:
                            upper_cej_pts.extend([m_cej, d_cej])
                            upper_crest_pts.extend([m_crest, d_crest])
                        elif 31 <= tooth_num <= 48:
                            lower_cej_pts.extend([m_cej, d_cej])
                            lower_crest_pts.extend([m_crest, d_crest])
                        else:
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
                            
                    draw_connected_line(upper_cej_pts, (255, 0, 0))
                    draw_connected_line(lower_cej_pts, (255, 0, 0))
                    draw_connected_line(upper_crest_pts, (255, 165, 0))
                    draw_connected_line(lower_crest_pts, (255, 165, 0))
                            
                st.image(overlay, caption="통합 시각화 (병소 탐지 + 치조골 레벨)")
                st.success("통합 분석 완료!")
