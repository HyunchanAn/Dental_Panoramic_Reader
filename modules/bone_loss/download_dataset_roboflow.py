import os
from roboflow import Roboflow

def download_roboflow_dataset():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("Roboflow에서 API Key를 발급받은 후 다음과 같이 실행해 주세요:")
        print("Windows (PowerShell): $env:ROBOFLOW_API_KEY='your_api_key'; python download_dataset_roboflow.py")
        print("Windows (CMD): set ROBOFLOW_API_KEY=your_api_key && python download_dataset_roboflow.py")
        return

    # Initialize Roboflow client
    rf = Roboflow(api_key=api_key)
    
    # Download ufba-425 dataset from Universe
    print("Downloading ufba-425 dataset...")
    project = rf.workspace("teeth-segmentation").project("ufba-425")
    version = project.version(1)
    # YOLOv8 format is 100% compatible with YOLOv11
    dataset = version.download("yolov8")
    
    print(f"Dataset successfully downloaded to: {dataset.location}")

if __name__ == "__main__":
    download_roboflow_dataset()
