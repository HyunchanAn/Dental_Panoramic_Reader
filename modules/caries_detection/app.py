import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import torch
import gc
from huggingface_hub import hf_hub_download
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from dentex_caries import CariesDetector, apply_clahe
# from src.explain import get_xai_heatmap  # This will be handled by CariesDetector.explain

st.set_page_config(page_title="Caries Detection AI", layout="wide")

st.title("🦷 Panoramic Caries Detection (우식 탐지)")
st.markdown("""
이 어플리케이션은 **YOLOv11** 모델을 사용하여 파노라마 X-ray 이미지에서 치아 우식(Caries)을 탐지합니다.
""")

# Sidebar for Model Selection
st.sidebar.header("Model Settings")
model_source = st.sidebar.radio("모델 선택", ["기본 모델 (yolo11s.pt)", "사용자 학습 모델"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "best_refined.pt") # Default to refined model
if model_source == "사용자 학습 모델":
    custom_model_path = st.sidebar.text_input("모델 경로 (.pt 파일)", "models/best_refined.pt")
    if os.path.exists(custom_model_path):
        model_path = custom_model_path
    else:
        st.sidebar.warning("지정된 경로에 모델이 없습니다. 기본 모델을 사용합니다.")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
st.sidebar.markdown("---")
st.sidebar.subheader("SAHI (Slicing) Settings")
use_sahi = st.sidebar.checkbox("Use SAHI (Sliced Inference)", value=False, help="작은 충치를 더 잘 찾기 위해 이미지를 조각내어 분석합니다.")
slice_size = st.sidebar.select_slider("Slice Size", options=[320, 640, 800, 1024], value=640, disabled=not use_sahi)
overlap_ratio = st.sidebar.slider("Overlap Ratio", 0.0, 0.5, 0.2, 0.05, disabled=not use_sahi)

st.sidebar.markdown("---")
st.sidebar.subheader("Image Preprocessing")
use_clahe = st.sidebar.checkbox("Apply CLAHE (Contrast Enhancement)", value=True, help="의료 영상의 명암비를 개선하여 작은 충치를 더 잘 보이게 합니다.")
clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 0.0, 5.0, 2.0, 0.5, disabled=not use_clahe)

st.sidebar.markdown("---")
st.sidebar.subheader("XAI Settings")
show_xai = st.sidebar.checkbox("Show XAI Heatmap (Explainable AI)", value=False, help="모델이 이미지의 어느 부분을 보고 판단했는지 히트맵으로 보여줍니다.")

st.sidebar.markdown("---")
line_width = st.sidebar.slider("Line Width (선 굵기)", 1, 5, 2)
font_size = st.sidebar.slider("Font Size (글자 크기)", 5, 50, 15)

@st.cache_resource
def ensure_model_exists(local_path, hf_repo="HyunchanAn/Caries_Detection_from_Panoramic", hf_filename="best_refined.pt"):
    if not os.path.exists(local_path):
        try:
            st.sidebar.info("Hugging Face에서 모델을 다운로드합니다...")
            downloaded_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
            return downloaded_path
        except Exception as e:
            st.sidebar.warning(f"HF 다운로드 실패: {e}")
            return local_path
    return local_path

@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

@st.cache_resource
def get_detector(path):
    return CariesDetector(model_path=path)

resolved_model_path = ensure_model_exists(model_path)
model = load_model(resolved_model_path)

# Main Interface
uploaded_file = st.file_uploader("파노라마 이미지 업로드", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='원본 이미지', use_container_width=True)

    if st.button("탐지 시작 (Detect)"):
        if model:
            with st.spinner("AI가 분석 중입니다..."):
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Image resizing for memory safety
                max_dim = 1024
                h, w = img_bgr.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
                
                detector = get_detector(resolved_model_path)
                final_boxes, processed_img_bgr = detector.predict(
                    img_bgr, 
                    use_clahe=use_clahe, 
                    clahe_clip=clahe_clip,
                    use_sahi=use_sahi, 
                    slice_size=slice_size, 
                    overlap_ratio=overlap_ratio,
                    conf=conf_threshold
                )
                processed_img_rgb = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)

                # --- Visualization ---
                res_image = Image.fromarray(processed_img_rgb)
                draw = ImageDraw.Draw(res_image, "RGBA")
                scale = max(1, res_image.width // 1000)
                font_size_score = int(font_size * (res_image.width / 1000) * 0.8)
                font_size_legend = int(14 * (res_image.width / 1000))

                import glob
                try:
                    font_paths = ["C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "arial.ttf"]
                    font_path = next((p for p in font_paths if os.path.exists(p)), None)
                    if not font_path:
                        for p in ["/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf"]:
                            potential = glob.glob(p, recursive=True)
                            for f in potential:
                                if any(k in f.lower() for k in ["nanum", "gothic", "malgun"]):
                                    font_path = f; break
                            if font_path: break
                    if font_path:
                        font_legend = ImageFont.truetype(font_path, font_size_legend)
                        font_score = ImageFont.truetype(font_path, font_size_score)
                    else:
                        font_legend = ImageFont.load_default(); font_score = ImageFont.load_default()
                except:
                    font_legend = ImageFont.load_default(); font_score = ImageFont.load_default()

                class_colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 165, 0), 3: (128, 0, 128)}

                for item in final_boxes:
                    bbox, conf, cls_id = item["bbox"], item["conf"], item["cls"]
                    color = class_colors.get(cls_id, (0, 0, 0))
                    draw.rectangle(bbox, outline=color, width=line_width)
                    # 클래스명 + 신뢰도만 표시
                    display_text = f"{item['name']} {conf:.2f}"
                    text_bbox = draw.textbbox((bbox[0], bbox[1]), display_text, font=font_score)
                    draw.rectangle([text_bbox[0], bbox[1] - font_size_score - 4, text_bbox[2] + 4, bbox[1]], fill=color)
                    text_color = (255, 255, 255) if cls_id == 0 else (0, 0, 0)
                    draw.text((bbox[0] + 2, bbox[1] - font_size_score - 2), display_text, font=font_score, fill=text_color)

                legend_items = [{"name": "Impacted (매복치)", "color": (0, 0, 255)}, {"name": "Caries (충치)", "color": (0, 255, 255)}, {"name": "Periapical Lep. (치근단)", "color": (255, 165, 0)}, {"name": "Deep Caries (깊은충치)", "color": (128, 0, 128)}]
                start_x, start_y, spacing = 15 * scale, 20 * scale, 22 * scale
                max_text_width = 0
                for item in legend_items:
                    text_bbox = draw.textbbox((0, 0), item["name"], font=font_legend)
                    max_text_width = max(max_text_width, text_bbox[2] - text_bbox[0])
                bg_w, bg_h = (20 * scale) + max_text_width + (15 * scale), len(legend_items) * spacing + 10 * scale
                draw.rectangle([start_x - 8, start_y - 12, start_x + bg_w, start_y + bg_h - 10], fill=(0, 0, 0, 80))
                for i, item in enumerate(legend_items):
                    y_pos = start_y + (i * spacing)
                    draw.rectangle([start_x, y_pos - 6 * scale, start_x + 12 * scale, y_pos + 6 * scale], fill=item["color"])
                    draw.text((start_x + 20 * scale, y_pos - 10 * scale), item["name"], font=font_legend, fill=(255, 255, 255))

            with col2:
                st.image(res_image, caption='분석 결과', use_container_width=True)
            
            st.subheader("탐지된 객체 목록")
            if final_boxes:
                for item in final_boxes:
                    st.write(f"- {item['name']}: {item['conf']:.2%}")
            else:
                st.info("탐지된 객체가 없습니다.")
            
            if show_xai:
                st.markdown("---")
                st.subheader("🔍 AI 판단 근거 분석 (XAI Heatmap)")
                st.warning("⚠️ 실험적 기능: 히트맵은 모델의 활성화 패턴을 시각화한 것으로, 정확한 병소 위치와 일치하지 않을 수 있습니다. 참고 목적으로만 사용하세요.")
                with st.spinner("히트맵을 생성 중입니다..."):
                    # Temporarily save image for explainability module
                    temp_img_path = "temp_xai.png"
                    image.save(temp_img_path) # Use original image
                    
                    viz, _ = detector.explain(temp_img_path)
                    if viz is not None:
                        st.image(viz, caption="Eigen-CAM Heatmap (붉은색일수록 모델이 집중한 영역입니다)", use_container_width=True)
                        st.info("💡 히트맵의 붉은 영역은 모델이 진단을 내릴 때 가장 중요하게 참고한 시각적 특징점들입니다.")
                    else:
                        st.warning("히트맵 생성에 실패했습니다.")
                
                # 가비지 컬렉션
                del img_array, img_bgr, processed_img_bgr
                gc.collect()
        else:
            st.error("모델이 로드되지 않았습니다.")

st.markdown("---")
st.markdown("Developed with YOLOv11 & Streamlit")
