import os
import sys

# Add Dental_Core to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules", "Dental_Core"))
from core.onnx_exporter import export_yolov8_to_onnx

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modules_dir = os.path.join(base_dir, "modules")
    
    # Define YOLO models to export
    yolo_models = [
        os.path.join(modules_dir, "Dental_002", "models", "best_refined.pt"),
        os.path.join(modules_dir, "Dental_008", "runs", "segment", "yolo_dentex", "yolov8m_seg_run", "weights", "best.pt"),
        os.path.join(modules_dir, "Dental_012", "models", "best.pt"),
        os.path.join(modules_dir, "Dental_013", "models", "best.pt")
    ]
    
    for pt_path in yolo_models:
        if os.path.exists(pt_path):
            print(f"\n--- Exporting {pt_path} ---")
            export_yolov8_to_onnx(pt_path)
        else:
            print(f"\n[SKIP] Not found: {pt_path}")

if __name__ == "__main__":
    main()
