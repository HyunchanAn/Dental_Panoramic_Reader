![Status](https://img.shields.io/badge/Status-v1.0%20Release-brightgreen) ![Python](https://img.shields.io/badge/Python-3.12%2B-blue) ![Backend](https://img.shields.io/badge/Backend-YOLOv8-red) ![UI](https://img.shields.io/badge/UI-Streamlit-orange) ![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD%20Pipeline-passing-brightgreen?logo=github)

# Dental Panoramic Reader

파노라마 방사선 사진(Panoramic Radiograph)을 입력받아 화질 개선부터 치아 식별, 병소 탐지, 결손치 파악, 매복치 분석, 치조골 소실량 측정까지 아우르는 **End-to-End 통합 진단 애플리케이션**입니다.
서로 다른 역할을 수행하는 인공지능 모듈(002, 003, 004, 008, 009, 010)을 서브모듈로 구성하고, 이를 하나의 일관된 진단 리포트로 통합(Orchestration)합니다.

## 핵심 파이프라인 (Architecture & Data Flow)

파이프라인은 데이터베이스나 외부 API 통신 없이 Python 런타임 내장 참조(In-memory Submodule Call)를 통해 최적화된 속도로 동작합니다.

```mermaid
graph TD
    %% Main Inputs
    Input["Raw Panoramic Image"] --> |Optional| SR("Dental_004: Super Resolution<br/>이미지 화질 개선")
    Input --> |Bypass| MainFlow
    SR --> MainFlow{"Image Ready"}

    %% Phase 1 & 2: Global Analysis & Segmentation
    MainFlow --> D8_Class("Dental_008_Classifier: Deciduous Check<br/>유치 식별")
    D8_Class --> |"has_deciduous"| D8_Seg("Dental_008: Tooth Segmentation<br/>치아 분할 및 FDI 부여")

    %% Phase 3: Detailed Prediction Modules
    D8_Seg --> |"Tooth ROI"| D2("Dental_002: Caries Detection<br/>우식/충치 탐지")
    D8_Seg --> |"Tooth ROI"| D12("Dental_012: Periapical Lesion<br/>치근단 병소 탐지")
    D8_Seg --> |"Tooth ROI"| D13("Dental_013: Restoration Predictor<br/>보철물 분류")
    D8_Seg --> |"Skip if Child"| D3("Dental_003: Bone Loss<br/>치조골 소실 측정")

    %% Post-processing & Output
    D2 --> OutputReport(("Final Report<br/>JSON & UI Render"))
    D3 --> OutputReport
    D12 --> OutputReport
    D13 --> OutputReport

    %% Resource Management
    subgraph "GPU Memory Orchestration"
        MM["Model Manager<br/>(OOM 방지 / 동적 Load & Unload)"] -.-> SR
        MM -.-> D8_Class
        MM -.-> D8_Seg
        MM -.-> D2
        MM -.-> D3
        MM -.-> D12
        MM -.-> D13
    end

    classDef optional fill:#f9f,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef core fill:#bbf,stroke:#333,stroke-width:2px,color:#000;
    classDef analysis fill:#bfb,stroke:#333,stroke-width:2px,color:#000;
    classDef output fill:#fbb,stroke:#333,stroke-width:2px,color:#000;

    class SR optional;
    class D8_Class,D8_Seg core;
    class D2,D3,D12,D13 analysis;
    class OutputReport output;
```

### 각 서브모듈(Submodule)별 상세 역할

1. **Model Manager (Orchestrator)**
   - VRAM(GPU 메모리) 초과를 방지하기 위해 각 서브모듈의 모델 가중치를 추론 시점에만 동적으로 GPU로 올리고(Load), 사용이 끝나면 내리는(Unload) 메모리 관리 관제탑입니다.

2. **Dental_004 (Super Resolution)** *[선택]*
   - 저화질 파노라마 사진이 입력되었을 때, 딥러닝 기반으로 해상도를 끌어올려 진단 정확도를 향상시킵니다.

3. **Dental_008 (Tooth Segmentation & Classification)** *[핵심/코어]*
   - **Classifier**: 사진의 치아 발달 상태를 보고 유치(Deciduous)가 포함되어 있는지 판별합니다. 유치일 경우 성인의 뼈 기준이 적용되는 003(치조골) 모듈 등의 실행을 스킵하도록 파이프라인을 제어합니다.
   - **Segmentation**: 치아 하나하나의 테두리(BBox)와 영역(Mask)을 따내고, 위치를 기반으로 국제 치아 식별 번호(FDI)를 부여합니다. 이 데이터는 뒤이어 실행되는 모든 분석 모듈의 나침반 역할을 합니다.

4. **Dental_002 (Caries Detection)**
   - 충치(우식) 병소를 탐지합니다. 여기서 탐지된 병소의 BBox 좌표가 008에서 찾은 치아 BBox와 겹치는지(IoU) 계산하여, 최종적으로 "몇 번 치아에 충치가 있는지"를 매핑합니다.

5. **Dental_003 (Bone Loss)**
   - 치아 잇몸뼈(치조골)가 얼마나 주저앉았는지(소실되었는지) 비율로 측정합니다. (유치가 감지되면 이 단계는 자동으로 건너뜁니다.)

6. **Dental_012 (Periapical Lesion)**
   - 008에서 제공한 치아 BBox를 중심으로 이미지를 크롭하여, 치아 뿌리 끝에 염증이나 병소(치근단 병소)가 있는지 정밀 탐지합니다.

7. **Dental_013 (Restoration Classification)**
   - 각 치아 ROI 이미지를 바탕으로 치아가 Crown, Implant, Filling, RCT(신경치료) 등 어떤 치료/수복을 받은 상태인지 EfficientNet-B0를 통해 5개 클래스로 분류합니다.

> ※ `.gitmodules`에 등록된 009(매복치), 010(결손/과잉치) 등은 차후 개발 및 고도화가 완료되는 대로 메인 흐름에 연동될 예정입니다.

## 설치 및 실행 방법

### 1. 소스코드 다운로드
Git Submodule을 포함하여 모든 모듈 코드를 다운로드합니다.
```bash
git clone --recursive https://github.com/HyunchanAn/Dental_Panoramic_Reader.git
cd Dental_Panoramic_Reader
```

### 2. 패키지 설치
각 서브모듈이 요구하는 라이브러리를 통합하여 설치합니다.
```bash
pip install -r requirements.txt
```

### 3. 애플리케이션 실행
```bash
streamlit run app.py
```

## 시스템 요구사항
- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM) 수준에 맞추어 `core/model_manager.py`가 구동 시나리오별 GPU 메모리 스왑 및 캐시 클리어링(PyTorch VRAM 최적화)을 자동 수행합니다.
- **OS**: Windows / Linux 지원

## 라이선스
MIT License


## 개요
이 레포지토리는 치과 AI 모듈러 시스템의 일부입니다.
