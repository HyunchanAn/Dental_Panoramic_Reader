# [ 261124 1624 랩탑 개발팀 ] (Dental_Panoramic_Reader#15 비공개 Hugging Face Space 토큰 준비 및 배포 런북)

## 1. 개요 및 선행 완료 상태

- 관련 이슈: [Dental_Panoramic_Reader#15](https://github.com/HyunchanAn/Dental_Panoramic_Reader/issues/15) (상위 총괄: [Dental_000#13](https://github.com/HyunchanAn/Dental_000/issues/13))
- 코드 및 컨테이너 선행 완료 항목 (커밋 `7d944c1`, 원격 CI 및 Docker 빌드 검증 완료):
  - CPU 전용 의존성 분리 (`requirements-cpu.txt`) 및 `requirements.txt` 정비
  - 서브모듈 자동 초기화 및 폴백 클론을 포함한 `Dockerfile`
  - 컨테이너 시작 시 `weights_manifest.yaml` 기반 가중치 자동 다운로드 엔트리포인트 (`docker_entrypoint.sh`)
  - 비공개 Hugging Face Space 배포 워크플로우 (`.github/workflows/deploy-hf-space.yml`)
  - Fail-Closed API 토큰 인증 (`DPR_API_TOKEN`) 및 인메모리 DICOM PHI 익명화

---

## 2. Hugging Face Write Token 발급 및 로컬 보안 주입 절차

1. [신규 Write 토큰 발급]:
   - https://huggingface.co/settings/tokens 에 접속한다.
   - 기존에 다른 프로젝트에서 사용 중인 토큰은 원문 재조회가 불가하며 `Refresh` 시 기존 연동이 해제되므로 건드리지 않고, 우측 상단 `[+ Create new token]`에서 `Write` 권한으로 전용 신규 토큰(예: `Dental_Panoramic_Reader_Space`)을 발급한다.
2. [로컬 안전 저장 (로그 노출 방지)]:
   - 토큰 원문은 버전 관리 파일이나 기록 로그에 남기지 않고, 로컬 전용(`127.0.0.1`) 일회성 입력 폼 또는 PowerShell `Read-Host -AsSecureString` 명령을 통해 로컬 사용자 환경 파일(`~/.env`의 `HF_TOKEN`) 및 Hugging Face 로컬 인증 저장소에만 등록한다.

---

## 3. 토큰 등록 후 배포 마감 4단계 체크리스트

- [ ] 1. `huggingface_hub.HfApi`를 통해 비공개 Docker Space (`<hf_username>/dental-panoramic-reader`, `private=True`, `space_sdk="docker"`) 생성
- [ ] 2. 해당 Space Secrets에 `DPR_API_TOKEN` (강력한 난수 문자열) 및 `DPR_CORS_ORIGINS` (허용 프론트엔드 URL 목록) 등록
- [ ] 3. `HyunchanAn/Dental_Panoramic_Reader` GitHub Repository Secrets에 `HF_TOKEN` 및 `HF_SPACE_REPO` 등록 (`gh secret set`)
- [ ] 4. `.github/workflows/deploy-hf-space.yml` 실행 후 컨테이너 정상 빌드·기동 확인 및 `Dental_Panoramic_Reader#15`·`Dental_000#13` 종결
