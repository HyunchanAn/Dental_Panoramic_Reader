import cv2
import os
from ultralytics import YOLO
import numpy as np
import pandas as pd

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

def extract_fp_candidates(model_path, image_dir, label_dir, class_names, iou_threshold=0.3, conf_threshold=0.3):
    model = YOLO(model_path)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    fp_candidates = []
    
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cls_id, cx, cy, nw, nh = map(float, parts)
                    gt_boxes.append({"box": [int((cx-nw/2)*w), int((cy-nh/2)*h), int((cx+nw/2)*w), int((cy+nh/2)*h)], "cls": int(cls_id)})
        
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        pred_boxes = [{"box": b.xyxy[0].tolist(), "cls": int(b.cls[0]), "conf": float(b.conf[0])} for b in results[0].boxes]
            
        for p in pred_boxes:
            is_fp = True
            for g in gt_boxes:
                # If it overlaps with ANY ground truth box, it might be a localization error or same object
                if calculate_iou(p["box"], g["box"]) > iou_threshold:
                    is_fp = False
                    break
            
            if is_fp:
                fp_candidates.append({
                    "image": img_name,
                    "class": class_names[p["cls"]],
                    "conf": p["conf"],
                    "box": p["box"]
                })
                
    df = pd.DataFrame(fp_candidates)
    if not df.empty:
        df = df.sort_values(by="conf", ascending=False)
    return df

if __name__ == "__main__":
    CLASSES = {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"}
    MODELS_PT = "models/best.pt"
    BASE_DIR = "data/processed"
    
    all_dfs = []
    for split in ["train", "val"]:
        img_dir = os.path.join(BASE_DIR, split, "images")
        lbl_dir = os.path.join(BASE_DIR, split, "labels")
        if os.path.exists(img_dir):
            print(f"Processing {split} split...")
            df = extract_fp_candidates(MODELS_PT, img_dir, lbl_dir, CLASSES)
            df["split"] = split
            all_dfs.append(df)
            
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("fp_candidates.csv", index=False)
    print(f"Extracted {len(final_df)} FP candidates to fp_candidates.csv")
    if not final_df.empty:
        print(final_df.head(20))
