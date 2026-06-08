# 구현 계획서 (Implementation Plan) - Panoramic Caries Detection

## 목표 (Goal)
치과 파노라마 X-ray 이미지에서 우식(Caries)을 탐지하는 YOLOv11 기반의 딥러닝 모델을 구축하고, 이를 Streamlit 웹 애플리케이션으로 서비스합니다.

## 사용자 리뷰 필요 사항 (User Review Required)
> [!IMPORTANT]
> **데이터셋 확보**: 모델 학습을 위해서는 DENTEX Challenge 2023 데이터셋이 필요합니다. 사용자가 직접 데이터를 다운로드하여 `data/raw` 폴더에 위치시켜야 합니다.
> **GPU 환경**: YOLO 학습은 GPU가 강력히 권장됩니다. 로컬 환경에 CUDA가 세팅되어 있는지 확인이 필요합니다.

## 제안된 변경 사항 (Proposed Changes)

### 1. 프로젝트 구조 (Project Structure)
- `app.py`: Streamlit 메인 애플리케이션 파일
- `train.py`: YOLOv11 모델 학습 스크립트
- `src/`
    - `data_converter.py`: DENTEX 데이터를 YOLO 포맷으로 변환
    - `inference.py`: 모델 로드 및 추론 로직
    - `visualizer.py`: 결과 이미지에 Bounding Box 시각화
- `data/`: 데이터 저장소 (Git ignore 처리)

### 2. 세부 구현 내용 (Detailed Implementation)

#### [NEW] [requirements.txt](file:///c:/Users/user/Documents/Github/Caries_Detection_from_Panoramic/requirements.txt)
- `ultralytics` (YOLOv11)
- `streamlit`
- `pillow`
- `pandas`
- `numpy`
- `opencv-python-headless`

#### [NEW] [app.py](file:///c:/Users/user/Documents/Github/Caries_Detection_from_Panoramic/app.py)
- Streamlit을 사용하여 사용자 인터페이스 구성
- 파일 업로더 구현
- Ultralytics 라이브러리를 통한 추론 실행 및 결과 표시

#### [NEW] [src/data_converter.py](file:///c:/Users/user/Documents/Github/Caries_Detection_from_Panoramic/src/data_converter.py)
- DENTEX 데이터셋의 Annotation 형식을 YOLO(`class x_center y_center width height`) 형식으로 변환

## 검증 계획 (Verification Plan)
### 자동화된 테스트
- 변환된 데이터셋의 좌표가 원본 이미지와 일치하는지 시각적으로 확인하는 스크립트 실행
- `app.py` 실행 후 샘플 이미지 업로드 시 에러 없이 박스가 쳐지는지 확인

### 수동 검증
- Streamlit 앱 로컬 실행 (`streamlit run app.py`)
- DENTEX 샘플 이미지 업로드 및 충치 탐지 확인
