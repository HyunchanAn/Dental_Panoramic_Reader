import os
import cv2
import numpy as np
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import json
from datetime import datetime
import sys

# Add current dir to path
sys.path.append(os.path.dirname(__file__))
from preprocess import apply_clahe

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / float(area1 + area2 - inter)

def evaluate_full_pipeline(model_path, img_dir, label_dir, class_names, use_sahi=True, use_clahe=True):
    """
    Evaluates the full pipeline (CLAHE + SAHI) manually to get precise metrics.
    """
    print(f"Starting Full Pipeline Evaluation (CLAHE={use_clahe}, SAHI={use_sahi})...")
    
    if use_sahi:
        # Load SAHI wrapper
        detection_model = UltralyticsDetectionModel(
            model_path=model_path,
            confidence_threshold=0.25,
            device="cuda:0"
        )
    else:
        model = YOLO(model_path)

    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    all_iou_values = []
    confusion_matrix = np.zeros((len(class_names)+1, len(class_names)+1)) # +1 for Background
    
    tp, fp, fn = {cid: 0 for cid in class_names.keys()}, {cid: 0 for cid in class_names.keys()}, {cid: 0 for cid in class_names.keys()}

    for img_name in image_files:
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if use_clahe:
            img = apply_clahe(img)
        h, w, _ = img.shape
        
        # Load GT
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, nw, nh = map(float, line.strip().split())
                    gt_boxes.append({"box": [(cx-nw/2)*w, (cy-nh/2)*h, (cx+nw/2)*w, (cy+nh/2)*h], "cls": int(cls), "matched": False})
        
        # Inference
        if use_sahi:
            result = get_sliced_prediction(img, detection_model, slice_height=640, slice_width=640, overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
            preds = [{"box": p.bbox.to_xyxy(), "cls": p.category.id} for p in result.object_prediction_list]
        else:
            result = model.predict(img, conf=0.25, verbose=False)
            preds = [{"box": b.xyxy[0].tolist(), "cls": int(b.cls[0])} for b in result[0].boxes]
            
        # Match
        for p in preds:
            best_iou, best_gt = 0.4, None # 0.4 threshold
            found_match = False
            for g in gt_boxes:
                iou = calculate_iou(p["box"], g["box"])
                if iou > best_iou:
                    best_iou, best_gt = iou, g
                    found_match = True
            
            if found_match:
                if p["cls"] == best_gt["cls"]:
                    tp[p["cls"]] += 1
                    best_gt["matched"] = True
                    all_iou_values.append(best_iou)
                    confusion_matrix[p["cls"]][p["cls"]] += 1
                else:
                    # Confusion
                    confusion_matrix[p["cls"]][best_gt["cls"]] += 1
            else:
                # False Positive (Predicted something not in GT)
                fp[p["cls"]] += 1
                confusion_matrix[p["cls"]][len(class_names)] += 1 # Background class
                
        for g in gt_boxes:
            if not g["matched"]:
                fn[g["cls"]] += 1
                confusion_matrix[len(class_names)][g["cls"]] += 1 # Background class

    # Summary
    print("\n" + "="*50)
    print(f"{'Class':20} | {'Prec':6} | {'Recall':6} | {'F1':6}")
    print("-" * 50)
    for cid, name in class_names.items():
        p = tp[cid] / (tp[cid] + fp[cid]) if (tp[cid] + fp[cid]) > 0 else 0
        r = tp[cid] / (tp[cid] + fn[cid]) if (tp[cid] + fn[cid]) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        print(f"{name:20} | {p:.3f}  | {r:.3f}   | {f1:.3f}")
    
    print("\nAverage IoU (TP only):", np.mean(all_iou_values) if all_iou_values else 0)
    
    # Save Results
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {"use_sahi": use_sahi, "use_clahe": use_clahe},
        "tp": tp, "fp": fp, "fn": fn,
        "avg_iou": float(np.mean(all_iou_values)) if all_iou_values else 0,
        "confusion_matrix": confusion_matrix.tolist()
    }
    with open("deep_verification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    return report

if __name__ == "__main__":
    evaluate_full_pipeline(
        model_path="models/best.pt",
        img_dir="data/processed/val/images",
        label_dir="data/processed/val/labels",
        class_names={0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"},
        use_sahi=True,
        use_clahe=True
    )
