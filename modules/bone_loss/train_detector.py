import os
from ultralytics import YOLO

def train():
    # 데이터셋 yaml 파일 경로 설정
    # ufba-425 다운로드 스크립트(download_dataset_roboflow.py) 실행 시
    # 생성되는 폴더(ufba-425-1) 안의 data.yaml 파일을 가리켜야 합니다.
    dataset_yaml = "ufba-425-1/data.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"ERROR: {dataset_yaml} 파일을 찾을 수 없습니다.")
        print("먼저 download_dataset_roboflow.py를 실행하여 데이터를 다운로드해 주세요.")
        return

    print("YOLOv11 모델 초기화 (사전 학습된 가중치 로드)...")
    # yolo11n.pt는 가장 가벼운 nano 모델입니다. 
    # VRAM이 충분한 환경(RTX 5080 16GB)이므로 정확도를 위해 yolo11m.pt 등 사용 가능
    model = YOLO("yolo11n.pt") 

    print("YOLOv11 학습을 시작합니다...")
    # RTX 5080의 16GB VRAM에 맞게 설정
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=1024,
        batch=16,
        device="cuda",
        project="models",
        name="detector_train",
        amp=True  # Mixed Precision 사용으로 VRAM 및 속도 최적화
    )
    
    print("학습이 완료되었습니다!")
    best_weights = os.path.join("models", "detector_train", "weights", "best.pt")
    print(f"최적의 가중치가 다음 경로에 저장되었습니다: {best_weights}")

if __name__ == "__main__":
    train()
