from huggingface_hub import HfApi

def upload_weights():
    # 1. 본인의 Hugging Face 토큰을 입력하세요. (또는 터미널에서 huggingface-cli login 실행)
    # 2. repo_id를 본인의 계정명/레포지토리명으로 변경하세요.
    repo_id = "chemahc94/pano-boneloss-weights"
    
    api = HfApi()
    
    # 레포지토리가 없다면 생성
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} 준비 완료.")
    except Exception as e:
        print(f"레포지토리 확인 중 오류: {e}")

    # 업로드할 파일 경로
    files_to_upload = [
        "runs/detect/models/detector_train/weights/best.onnx",
        "runs/detect/models/detector_train/weights/best.pt",
        "models/pano_classifier.onnx",
        "models/pano_classifier.pt"
    ]

    for file_path in files_to_upload:
        import os
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            print(f"업로드 중: {file_path} -> {filename}")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"업로드 완료: {filename}")
        else:
            print(f"경고: 파일을 찾을 수 없습니다 - {file_path}")

if __name__ == "__main__":
    upload_weights()
