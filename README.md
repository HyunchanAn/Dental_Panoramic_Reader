# AI Panoramic Radiograph Reader

파노라마 방사선 사진 기반의 치아 우식 탐지(Caries Detection) 모델과 치조골 소실량 분석(Bone Loss Measurement) 모델을 하나의 사용자 경험(UX)으로 묶은 통합 Streamlit 애플리케이션입니다.

## 기능 구성 (Features)

이 애플리케이션은 두 가지 주요 모듈을 서브모듈 형태로 포함하고 있습니다.
1. **우식 및 병소 탐지 (Caries Detection)**: YOLOv11 기반, 우식 및 치근단 병소 등 탐지. XAI Heatmap 지원.
2. **치조골 소실 측정 (Bone Loss Measurement)**: YOLOv11(치아 검출) + SAM(마스킹) 기반 랜드마크 추출 및 RBL(%) 계산.
3. **통합 분석 (Integrated Analysis)**: 두 모듈을 동시에 실행하여 한 화면에 합성.

## 설치 및 실행 방법

### 1. 소스코드 다운로드
Git Submodule을 포함하여 모든 코드를 다운로드합니다.
```bash
git clone --recursive https://github.com/HyunchanAn/AI_Panoramic_Radiograph_Reader.git
cd AI_Panoramic_Radiograph_Reader
```

### 2. 패키지 설치
로컬 서브모듈(Caries, BoneLoss)을 함께 설치합니다.
```bash
pip install -r requirements.txt
```

### 3. 애플리케이션 실행
```bash
streamlit run app.py
```

### 4. 도커 기반 배포 (Docker)
어디서나 구동 가능한 도커를 지원합니다.
```bash
docker build -t ai-pano-reader .
docker run -d -p 8501:8501 ai-pano-reader
```

## 아키텍처 및 의존성
- Model: YOLOv11 (Ultralytics), Segment Anything (SAM), MobileNetV3 (OOD)
- Framework: PyTorch, ONNX Runtime, FastAPI(Optional)
- UI: Streamlit

## 라이선스
MIT License
