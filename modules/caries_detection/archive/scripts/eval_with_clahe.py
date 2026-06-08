import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.dirname(__file__))
from preprocess import apply_clahe

def evaluate_with_preprocess(model_path, data_yaml, img_dir, preprocess_fn=None):
    """
    Evaluates the model on preprocessed images.
    """
    print(f"Starting evaluation with preprocessing on {img_dir}...")
    model = YOLO(model_path)
    
    # 1. Create temporary preprocessed directory
    temp_dir = "data/temp_preprocessed"
    temp_img_dir = os.path.join(temp_dir, "images")
    os.makedirs(temp_img_dir, exist_ok=True)
    
    # Copy labels (symlink or copy)
    label_dir = img_dir.replace("images", "labels")
    temp_label_dir = os.path.join(temp_dir, "labels")
    if not os.path.exists(temp_label_dir):
        # On Windows, symlink might need admin, so we'll just copy or assume path
        import shutil
        shutil.copytree(label_dir, temp_label_dir, dirs_exist_ok=True)

    # Preprocess and save images
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for img_name in image_files:
        img = cv2.imread(os.path.join(img_dir, img_name))
        if preprocess_fn:
            img = preprocess_fn(img)
        cv2.imwrite(os.path.join(temp_img_dir, img_name), img)
        
    # 2. Create temporary data.yaml
    temp_yaml = os.path.join(temp_dir, "temp_data.yaml")
    with open(data_yaml, 'r') as f:
        lines = f.readlines()
    
    with open(temp_yaml, 'w') as f:
        for line in lines:
            if line.startswith("val:"):
                # Point to absolute path of temp images
                abs_path = os.path.abspath(temp_img_dir)
                f.write(f"val: {abs_path}\n")
            else:
                f.write(line)
                
    # 3. Run Validation
    results = model.val(data=temp_yaml, split='val', verbose=True)
    
    # 4. Save and Report
    metrics = {
        "mAP50": results.results_dict['metrics/m_ap50'],
        "mAP50-95": results.results_dict['metrics/m_ap'],
        "precision": results.results_dict['metrics/precision(b)'],
        "recall": results.results_dict['metrics/recall(b)'],
        "fitness": results.fitness
    }
    
    report_name = f"evaluation_clahe_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_name, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nEvaluation Results with CLAHE:")
    print(json.dumps(metrics, indent=4))
    print(f"Full report saved to: {report_name}")
    
    return metrics

if __name__ == "__main__":
    evaluate_with_preprocess(
        model_path="models/best.pt",
        data_yaml="data.yaml",
        img_dir="data/processed/val/images",
        preprocess_fn=apply_clahe
    )
