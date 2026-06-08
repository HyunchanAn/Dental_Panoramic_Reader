# 작업 이관 가이드 (HANDOFF NOTES)

이 파일은 **다른 컴퓨터로 작업을 이동**할 때 필요한 절차를 설명합니다.

## 1. 프로젝트 가져오기
- GitHub에 코드를 Push한 후, 새 컴퓨터에서 Pull 합니다.
- 또는 이 폴더 전체를 압축하여 이동합니다.

## 2. 환경 설정
새 컴퓨터에서 다음 명령어를 실행하여 필수 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

## 3. 데이터셋 다운로드 (가장 중요)
이전 컴퓨터에서는 저장 공간 부족으로 데이터를 받지 못했습니다. 새 컴퓨터(또는 외장 하드)에 충분한 공간(약 15GB 이상 권장)을 확보한 후 다음을 실행하세요.

**방법 A: 기본 경로 (`data/raw/DENTEX`)에 다운로드**
```bash
python src/download_data.py
```

**방법 B: 특정 경로(예: 외장 하드)에 다운로드**
```bash
python src/download_data.py --dir "D:\BigData\DENTEX"
```

## 4. 데이터 전처리
데이터 다운로드가 완료되면, 압축을 풀고 YOLO 형식으로 라벨을 변환해야 합니다.
1. `data/raw/train_quadrant_enumeration_disease.json` 파일이 있는지 확인합니다.
2. `src/data_converter.py` 파일을 열어 `if __name__ == "__main__":` 부분의 경로(`JSON_PATH`, `IMAGE_DIR`)를 실제 데이터 위치에 맞게 수정합니다.
3. 변환 스크립트 실행:
```bash
python src/data_converter.py
```

## 5. 학습 및 실행
데이터 준비가 끝나면 다음 단계로 넘어갑니다.
- **모델 학습**: `python train.py` (GPU 권장)
- **앱 실행**: `streamlit run app.py`

## 현재 진행 상황 요약
- [x] Streamlit UI 구현 (`app.py`)
- [x] YOLOv11 학습 스크립트 구현 (`train.py`)
- [x] 데이터 다운로드 도구 구현 (`src/download_data.py`)
- [ ] **데이터 다운로드 및 전처리 (대기 중)**
- [ ] 모델 학습 (대기 중)
