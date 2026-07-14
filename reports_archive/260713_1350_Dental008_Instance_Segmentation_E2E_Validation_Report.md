# Dental_008 (치아 인스턴스 분할 및 FDI 식별) - E2E Validation Report (NMS 패치 후)

- **작성일**: 2026-07-13 14:11 (GMT+9)
- **작성자**: Antigravity
- **검증 대상**: Dental_008 (Mask R-CNN 기반 Instance Segmentation)
- **검증 환경**:
  - OS: Windows 11
  - Python: 3.11
  - GPU: NVIDIA GeForce RTX 4060 Laptop GPU (또는 RTX 5080)
  - PyTorch: 2.1.0+cu121 (가정)

---

## 1. 개요 (Executive Summary)

과다 예측(False Positive) 현상을 개선하기 위해 **NMS (IoU 0.3) 및 Confidence 상향(0.75)** 패치를 적용한 뒤, Detection(치아 탐지 여부)과 Classification(FDI 번호 판별)을 분리하여 재평가를 진행했습니다.

### 벤치마크 평가 결과 (NMS 패치 후)
- **Inference Speed**: 0.1266 s/image (약 7.90 FPS)
- **Bounding Box IoU (TP)**: 0.8371
- **Mask IoU (TP)**: 0.8099

#### 1) Class-Agnostic (순수 치아 탐지 성능)
- **Recall**: **89.56%** (TP: 163, FN: 19)
- **Precision**: **12.58%** (FP: 1133)
- **F1 Score**: **0.2206**

#### 2) Class-Aware (치아 탐지 + FDI 분류 성능)
- **Recall**: **86.81%** (TP: 158, FN: 24)
- **Precision**: **12.19%** (FP: 1138)
- **F1 Score**: **0.2138**

**결론**: 패치 이후에도 **이미지 장당 평균 22개 이상의 고신뢰도(Confidence > 0.75) False Positive 박스**가 서로 겹치지 않게(IoU < 0.3) 발생하고 있습니다. FDI 번호를 틀려서 발생한 FP가 아니라, 아예 허공이나 배경을 치아로 강하게 오인하고 있는 현상입니다. 현재 불러온 가중치(`mask_rcnn_dentex_epoch_30.pth`) 자체가 충분히 수렴하지 못했거나, 학습 데이터셋 구성/하이퍼파라미터에 문제가 있었을 확률이 매우 높습니다. 이 상태로는 후처리(NMS)만으로 오진을 막을 수 없습니다. 추가적인 재학습(Retraining) 또는 모델 앙상블이 필요합니다.

---

## 2. 통합 아키텍처 (System Architecture)

```mermaid
graph TD
    A[입력 이미지 (Panoramic X-ray)] --> B[Dental_008 모듈]
    B --> C{Mask R-CNN 추론}
    C -->|BBox & Confidence > 0.75| D[NMS 필터링 (IoU 0.3)]
    C -->|Pixel Mask| E[Instance Mask 획득]
    D --> F[최종 Bounding Box 확정]
    E --> G[FDI 번호별 픽셀 폴리곤 산출]
    
    F --> H(후속 모듈: Dental_009 매복치 분석)
    G --> H
    G --> I(후속 모듈: Dental_010 결손치 분석 - 현재 FP 과다로 보류 권장)
    
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:1px
    style H fill:#dfd,stroke:#333,stroke-width:2px
    style I fill:#dfd,stroke:#333,stroke-width:2px
```

---

## 3. 실측 파노라마 E2E 추론 결과 (Real Inference)

추론 중 식별된 치열 패턴별 시각화 이미지는 `reports_archive/images` 폴더 내에 저장되었습니다.

* `eval_permanent.jpg`: 영구치(11~48) 탐지 사례
* `eval_deciduous.jpg`: 유치(51~85) 탐지 사례
* `eval_mixed.jpg`: 혼합치열 탐지 사례
