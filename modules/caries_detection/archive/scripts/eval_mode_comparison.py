"""
SAHI 없이 Baseline vs CLAHE만 빠르게 비교하는 경량 평가 스크립트.
SAHI 모드는 CPU에서 수십 분 소요되므로 별도 실행 권장.
"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from preprocess import apply_clahe

MODEL_PATH = "models/best.pt"
IMG_DIR = "data/processed/val/images"
LABEL_DIR = "data/processed/val/labels"
CLASS_NAMES = {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"}
IOU_THRESHOLD = 0.4
CONF_THRESHOLD = 0.25

def calc_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / float(a1+a2-inter)

def run_eval(img_dir, label_dir, use_clahe=False):
    model = YOLO(MODEL_PATH)
    images = [f for f in os.listdir(img_dir) if f.endswith(('.png','.jpg','.jpeg'))]
    tp = {k: 0 for k in CLASS_NAMES}
    fp = {k: 0 for k in CLASS_NAMES}
    fn = {k: 0 for k in CLASS_NAMES}

    for img_name in images:
        img = cv2.imread(os.path.join(img_dir, img_name))
        if use_clahe:
            img = apply_clahe(img)
        h, w = img.shape[:2]

        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0]+".txt")
        gts = []
        if os.path.exists(label_path):
            for line in open(label_path):
                c,cx,cy,nw,nh = map(float, line.strip().split())
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

def print_table(title, results):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  {'Class':18} | {'Prec':6} | {'Recall':6} | {'F1':6}")
    print(f"  {'-'*50}")
    for name, m in results.items():
        print(f"  {name:18} | {m['Precision']:.3f}  | {m['Recall']:.3f}   | {m['F1']:.3f}")

if __name__ == "__main__":
    print(f"\n빠른 모드 비교 시작 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print("(SAHI 평가는 CPU에서 시간 소요가 크므로 제외)")

    print("\n[1/2] Baseline (CLAHE off)...")
    r1 = run_eval(IMG_DIR, LABEL_DIR, use_clahe=False)
    print_table("Baseline", r1)

    print("\n[2/2] CLAHE only...")
    r2 = run_eval(IMG_DIR, LABEL_DIR, use_clahe=True)
    print_table("CLAHE only", r2)

    print("\n\n[Caries 성능 요약]")
    print(f"  {'지표':12} | {'Baseline':10} | {'CLAHE':10} | {'변화':10}")
    print(f"  {'-'*50}")
    for metric in ["Precision","Recall","F1"]:
        base = r1['Caries'][metric]; clahe = r2['Caries'][metric]
        diff = clahe - base
        sign = "+" if diff >= 0 else ""
        print(f"  {metric:12} | {base:.3f}      | {clahe:.3f}      | {sign}{diff:.3f}")
    print(f"\n완료.")
