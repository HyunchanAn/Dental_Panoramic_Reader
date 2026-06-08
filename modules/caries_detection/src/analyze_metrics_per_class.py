import os
import pandas as pd
from ultralytics import YOLO

def analyze_class_metrics(model_path, data_yaml):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Ensure it is downloaded.")
        return
        
    model = YOLO(model_path)
    print("Evaluating model to calculate per-class metrics...")
    metrics = model.val(data=data_yaml, split='val', save_json=False)
    
    print("\n--- Per-Class Performance (PR Curve metrics) ---")
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10}")
    print("-" * 60)
    
    class_names = model.names
    try:
        ap_class_index = metrics.box.ap_class_index
        for i, c in enumerate(ap_class_index):
            name = class_names[c]
            p = metrics.box.p[i]
            r = metrics.box.r[i]
            ap50 = metrics.box.map50[i]
            print(f"{name:<20} | {p:<10.4f} | {r:<10.4f} | {ap50:<10.4f}")
    except Exception as e:
        print(f"Could not parse per-class metrics: {e}")

if __name__ == "__main__":
    analyze_class_metrics("models/best_refined.pt", "data_refined.yaml")
