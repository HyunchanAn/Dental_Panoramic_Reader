import cv2
import os
from ultralytics import YOLO
import numpy as np

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)

def analyze_errors(model_path, image_dir, label_dir, output_dir, class_names, iou_threshold=0.4, conf_threshold=0.25):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    stats = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_names.keys()}
    
    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls_id, cx, cy, nw, nh = map(float, line.strip().split())
                    gt_boxes.append({"box": [int((cx-nw/2)*w), int((cy-nh/2)*h), int((cx+nw/2)*w), int((cy+nh/2)*h)], "cls": int(cls_id), "matched": False})
        
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        pred_boxes = [{"box": b.xyxy[0].tolist(), "cls": int(b.cls[0]), "conf": float(b.conf[0]), "matched": False} for b in results[0].boxes]
            
        for p in pred_boxes:
            best_iou, best_gt = 0, None
            for g in gt_boxes:
                if p["cls"] == g["cls"]:
                    iou = calculate_iou(p["box"], g["box"])
                    if iou > best_iou: best_iou, best_gt = iou, g
            
            if best_iou >= iou_threshold:
                p["matched"], best_gt["matched"] = True, True
                stats[p["cls"]]["tp"] += 1
                if idx < 20: # Draw only first 20
                    cv2.rectangle(img, (int(p["box"][0]), int(p["box"][1])), (int(p["box"][2]), int(p["box"][3])), (0, 255, 0), 2)
                    cv2.putText(img, f"TP: {class_names[p['cls']]}", (int(p['box'][0]), int(p['box'][1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                stats[p["cls"]]["fp"] += 1
                if idx < 20:
                    cv2.rectangle(img, (int(p["box"][0]), int(p["box"][1])), (int(p["box"][2]), int(p["box"][3])), (255, 0, 0), 2)
                    cv2.putText(img, f"FP: {class_names[p['cls']]}", (int(p['box'][0]), int(p['box'][1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
        for g in gt_boxes:
            if not g["matched"]:
                stats[g["cls"]]["fn"] += 1
                if idx < 20:
                    cv2.rectangle(img, (int(g["box"][0]), int(g["box"][1])), (int(g["box"][2]), int(g["box"][3])), (0, 0, 255), 2)
                    cv2.putText(img, f"FN: {class_names[g['cls']]}", (int(g['box'][0]), int(g['box'][1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if idx < 20: cv2.imwrite(os.path.join(output_dir, f"error_{img_name}"), img)

    print("\n" + "="*40)
    print(f"{'Class':20} | {'TP':4} | {'FP':4} | {'FN':4} | {'Recall':6}")
    print("-" * 40)
    for cid, name in class_names.items():
        s = stats[cid]
        recall = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
        print(f"{name:20} | {s['tp']:4} | {s['fp']:4} | {s['fn']:4} | {recall:.3f}")
    print("="*40)

if __name__ == "__main__":
    analyze_errors("models/best.pt", "data/processed/val/images", "data/processed/val/labels", "error_analysis", {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"})
