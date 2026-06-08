# 작업 목록 (Task List) - Panoramic Caries Detection

## 1. 프로젝트 초기화 (Project Initialization)
- [x] `README.md` 작성 (프로젝트 설명, 참고 문헌 정리)
- [x] `requirements.txt` 작성 (필수 라이브러리 명시)
- [x] 기본 디렉토리 구조 생성 (`data`, `models`, `src`, `utils`)

## 2. 데이터셋 준비 도구 (Data Preparation Tools)
- [x] DENTEX 데이터셋 다운로드 스크립트 작성 (`src/download_data.py`)
- [x] DENTEX Challenge 데이터셋 구조 분석
- [x] 포맷 변환 스크립트 작성 (`prepare_data.py`: COCO text/json -> YOLO txt)
- [ ] 데이터 분할 스크립트 (Train/Val/Test Split)

## 3. 모델 학습 환경 구축 (Model Training Setup)
- [x] YOLOv11 학습 스크립트 작성 (`train.py`)
- [x] 학습 설정 파일 작성 (`data.yaml`, `config.yaml`)
- [ ] 사전 학습된 가중치(Pre-trained weights) 다운로드 및 로드 테스트

## 4. 스트림릿 UI 개발 (Streamlit UI Development)
- [x] `app.py` 기본 골격 작성
- [x] 이미지 업로드 기능 구현 (`jpg`, `png` 지원)
- [x] 모델 추론 및 결과 시각화 (Bounding Box 그리기)
- [x] 사용자 설정 옵션 추가 (Confidence Threshold 조절 등)

## 5. 테스트 및 검증 (Testing & Verification)
- [ ] 샘플 이미지를 통한 추론 테스트
- [ ] 성능 지표 확인 방법 문서화
