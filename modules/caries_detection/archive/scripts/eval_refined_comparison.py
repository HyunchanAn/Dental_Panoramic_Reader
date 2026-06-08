import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Paths
ORIGINAL_MODEL = "models/best.pt"
REFINED_MODEL = "runs/detect/dentex_yolov11s_refined_v3/weights/best.pt"
REFINED_VAL_IMG = "data/refined/val/images"
REFINED_VAL_LBL = "data/refined/val/labels"
CLASS_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}
IOU_THRESHOLD = 0.4
CONF_THRESHOLD = 0.25

def calc_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / float(a1+a2-inter)

def run_eval(model_path, img_dir, label_dir):
    model = YOLO(model_path)
    images = [f for f in os.listdir(img_dir) if f.endswith(('.png','.jpg','.jpeg'))]
    tp = {k: 0 for k in CLASS_NAMES}
    fp = {k: 0 for k in CLASS_NAMES}
    fn = {k: 0 for k in CLASS_NAMES}

    for img_name in images:
        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is None: continue
        h, w = img.shape[:2]

        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0]+".txt")
        gts = []
        if os.path.exists(label_path):
            for line in open(label_path):
                parts = line.strip().split()
                if len(parts) != 5: continue
                c,cx,cy,nw,nh = map(float, parts)
                gts.append({"box":[(cx-nw/2)*w,(cy-nh/2)*h,(cx+nw/2)*w,(cy+nh/2)*h],"cls":int(c),"matched":False})

        result = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
        preds = [{"box":b.xyxy[0].tolist(),"cls":int(b.cls[0])} for b in result[0].boxes]

        for p in preds:
            best_iou, best_gt = IOU_THRESHOLD, None
            for g in gts:
                iou = calc_iou(p["box"], g["box"])
                if iou > best_iou:
                    best_iou, best_gt = iou, g
            if best_gt and p["cls"] == best_gt["cls"]:
                tp[p["cls"]] += 1; best_gt["matched"] = True
            else:
                fp[p["cls"]] += 1
        for g in gts:
            if not g["matched"]: fn[g["cls"]] += 1

    results = {}
    for cid, name in CLASS_NAMES.items():
        prec = tp[cid]/(tp[cid]+fp[cid]) if (tp[cid]+fp[cid]) > 0 else 0
        rec  = tp[cid]/(tp[cid]+fn[cid]) if (tp[cid]+fn[cid]) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        results[name] = {"Precision": round(prec,3), "Recall": round(rec,3), "F1": round(f1,3)}
    return results

def print_summary(res_base, res_ref):
    print("\n" + "="*70)
    print(f"{'Class':20} | {'Metric':10} | {'Base':8} | {'Refined':8} | {'Diff'}")
    print("-" * 70)
    for name in CLASS_NAMES.values():
        for m in ["Precision", "Recall", "F1"]:
            b = res_base[name][m]
            r = res_ref[name][m]
            d = r - b
            print(f"{name:20} | {m:10} | {b:.3f}    | {r:.3f}    | {'+' if d>=0 else ''}{d:.3f}")
        print("-" * 70)

if __name__ == "__main__":
    if not os.path.exists(REFINED_MODEL):
        print(f"Error: Refined model not found at {REFINED_MODEL}")
        sys.exit(1)
        
    print(f"Evaluating Baseline on Refined Val Set...")
    res_base = run_eval(ORIGINAL_MODEL, REFINED_VAL_IMG, REFINED_VAL_LBL)
    
    print(f"Evaluating Refined Model on Refined Val Set...")
    res_ref = run_eval(REFINED_MODEL, REFINED_VAL_IMG, REFINED_VAL_LBL)
    
    print_summary(res_base, res_ref)
