# [ 261125 1530 랩탑 개발팀 ] (Dental_Panoramic_Reader#21/#22/#15 통합 인터페이스·API 계약·배포 실행 계획서)

## 1. 대상 이슈 및 위치

- 대상 이슈:
  - [Dental_Panoramic_Reader#21](https://github.com/HyunchanAn/Dental_Panoramic_Reader/issues/21) (`[P0][통합] 개별 모듈 인터페이스·가중치 매니페스트 단절 수정`) -> `[READY]`
  - [Dental_Panoramic_Reader#22](https://github.com/HyunchanAn/Dental_Panoramic_Reader/issues/22) (`[P1][API/문서] Dental_015 API 응답 스키마·포트/CORS 정합화 및 README 전면 현행화`) -> `[READY]`
  - [Dental_Panoramic_Reader#15](https://github.com/HyunchanAn/Dental_Panoramic_Reader/issues/15) (`[FREEZE/승인대기][배포] 웹 조회용 배포 준비`) -> `[FREEZE]` (별도 런북: `plans_replies_walkthrough/261124_1624_HF_Private_Space_Token_and_Deploy_Runbook.md`)
- 상위 Umbrella: [Dental_000#18](https://github.com/HyunchanAn/Dental_000/issues/18) (`Dental_000/plans_replies_walkthrough/261125_1530_Dental_Series_Remediation_Umbrella_Plan.md` Track 3)
- 선행 권장: Track 1 (`Dental_Core#7`) 및 Track 2 (`Dental_009/010/012/013`)

---

## 2. 즉시 실행 가능 작업 (`[READY]`)

### 2.1 `#21` [P0] 개별 모듈 인터페이스 및 `weights_manifest.yaml` 단절 수정
1. `Dental_004` (`core/interfaces/dental_004.py:72-99`): 미존재 `PanoTiler.extract_tiles` 호출 제거(`tile_image`/`process_large_image` 정합화), `merge_tiles` 인자 수정, 2D grayscale -> 3채널 변환으로 직후 `dental_008.py:69` `ValueError` 크래시 해소.
2. `Dental_013` (`modules/restoration_predictor.py:13-101`): 실제 `Dental_013` 7-class EfficientNet-B0 치아 크롭 분류기(224x224)를 31-class YOLO-seg로 잘못 래핑한 구조 및 `self.session` ONNX 추론 누락(침묵 실패 `results: []`) 수정.
3. `Dental_011` (`core/pipeline.py:119-128`, `weights_manifest.yaml:59-63`, `modules/age_predictor.py:13-24`): `best_hybrid_age_model.onnx`와 `pipeline.py` 탐색명(`age_resnet50.onnx`/`best_model.pth`) 통일 및 `.pth` -> ONNX Runtime 로드 크래시 방지.
4. `Dental_003` (`core/interfaces/dental_003.py:61-65`, `weights_manifest.yaml:27-31`, `modules/boneloss_predictor.py:9-71`): SAM `.onnx`를 PyTorch `sam_model_registry`로 로드 시도하는 포맷 비호환 수정, `modules/Dental_003` `sys.path` 및 `xyxy -> xywh` 변환 추가.
5. `Dental_008` (`core/pipeline.py:192-203`, `core/interfaces/dental_008.py:64-72`, `modules/segmentation_predictor.py:60-64`): `model_008 is None` 가드 추가, `assign_fdi_labels()` 호출 누락 수정, `weights_manifest.yaml` 중복 정의 정리.
6. `Dental_009` (`core/pipeline.py:361-383`): 축정렬 BBox 4점 대신 `tooth_roi_data["contours"]` 폴리곤을 `analyze_impacted_tooth()`에 전달.
7. `Dental_010` (`modules/dental_010_heuristic.py:74-78`, `modules/missing_tooth_predictor.py:5-10`): 중절치 근심 인접치 계산 버그(`missing_fdi + 10`)를 최신 `Dental_010` `contralateral_q` 로직으로 동기화 및 `modules/Dental_Core` `sys.path` 추가.
8. `Dental_002` (`weights_manifest.yaml:12-21`, `core/interfaces/dental_002.py:30-58,133-148`): `best_refined.onnx -> best_patch.onnx` 저장으로 인한 patch 모드 오판 해소 및 `Periapical` 클래스 필터링 정합화.
9. `Dental_014` (`modules/osteoporosis_predictor.py:63`): OpenCV BGR -> RGB 변환 후 `PIL.Image.fromarray()` 전달.
10. 유틸리티 스크립트: `scripts/export_all_yolo_to_onnx.py:5-6` (`dental_core.core.onnx_exporter` 경로 수정 및 비-YOLO `Dental_013` 제외), `scripts/benchmark.py:43-55` (`np.ndarray` 입력 및 `'module_execution'` 키 수정), `scripts/export_classifier.py:8-9` 로컬 절대경로 제거.

### 2.2 `#22` [P1] API 응답 스키마·포트/CORS 정합화 및 `README.md` 전면 현행화
- `api_server.py:540-546`, `modules/dental_010_heuristic.py:29-43`: `findings.missingTeeth` 하위 키 `details` 통일(`gap_details` 제거) 및 8개 미만 검출 시 `details[].fdi` 필수 타입 방어.
- `api_server.py:365-378,463`: `findings.caries[]`에 `clinicalStatus` 반환 추가, `findings.restorations[].toothNumber` `int` 캐스팅 보장.
- `core/pipeline.py:180-182,197-203,446-447`, `api_server.py:236-310`: `moduleExecutionStatus`에 `Dental_004` 반영, `predicted_age: None` 및 `FDI_UNAVAILABLE` 시 `executed` 오표시 모순 수정.
- `docker_entrypoint.sh:10`, `run_server.py:6`, `api_server.py:43-60`: 포트(`PORT=7860` vs `API_PORT=8000`) 정책 문서화 및 `DEFAULT_ALLOWED_ORIGINS`/CORS 폴백 목록에 `3000`, `7860` 포트 추가.
- `README.md`: `8개 모듈` vs `9개 모듈` 내부 모순 수정, Docker/HF Space(`PORT=7860`, `docker_entrypoint.sh`, `weights_manifest.yaml`, `requirements-cpu.txt`) 반영, `Dental_011` 가중치명 일치화, `Dental_000/reports_archive/` 경로 정정, 로컬 `pytest tests -v` 및 `Python-3.11+` 배지 정합화.

---

## 3. `[FREEZE]` 보류 항목 (오너 승인 및 외부 설정 대기)

| 항목 | 상태 | 동결 사유 및 오너(안현찬) 역할 |
| :--- | :--- | :--- |
| `#15` 비공개 Hugging Face Space 생성, 토큰 주입 및 원격 실배포 | `[FREEZE / 오너 승인 필요]` | `261124_1624_HF_Private_Space_Token_and_Deploy_Runbook.md` 절차에 따라 오너(안현찬)가 신규 Write `HF_TOKEN` 발급 및 로컬 보안 주입, `DPR_API_TOKEN` / `DPR_CORS_ORIGINS` 시크릿 정책 결정, `#21`/`#22`/`Dental_015#4` 완료 확인 후 최종 배포를 승인해야 함. |
| (연계 확인) DICOM 프리뷰 이미지 반환 API 확장 여부 | `[FREEZE / 오너 승인 필요]` | `Dental_015#4`의 `.dcm` 브라우저 렌더링 방안으로 서버 프리뷰 반환(안 1)을 채택할 경우 `api_server.py` 응답 스키마에 프리뷰 필드를 추가해야 하므로 오너의 아키텍처 선택 대기. |

---

## 4. 검증 기준

- 로컬 검증: `pytest tests -v` (기존 `tests/test_api_and_pipeline.py` + 각 모듈 인터페이스 모킹/단위 테스트 및 API 스키마 회귀 테스트 추가) 전수 통과.
- 제약 명시: 로컬에 실제 학습 가중치(`.onnx`/`.pth`) 파일이 없는 모듈은 실모델 추론 대신 인터페이스·매니페스트·입출력 텐서 형태 검증 테스트로 보증.
- 원격 검증: GitHub Actions CI 통과 확인.
