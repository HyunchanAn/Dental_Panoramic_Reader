import cv2
import os
import numpy as np

def verify_refined(image_dir, label_dir, output_dir, class_names, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    
    # Filter only those that were likely modified (or just take some)
    # To be sure we see refined ones, let's look for images that had many boxes added
    
    count = 0
    for lbl_name in label_files:
        lbl_path = os.path.join(label_dir, lbl_name)
        img_name = os.path.splitext(lbl_name)[0] + ".png"
        img_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(img_path):
            # check jpg
            img_path = os.path.join(image_dir, os.path.splitext(lbl_name)[0] + ".jpg")
            if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5: continue
                cls_id, cx, cy, nw, nh = map(float, parts)
                x1 = int((cx - nw/2) * w)
                y1 = int((cy - nh/2) * h)
                x2 = int((cx + nw/2) * w)
                y2 = int((cy + nh/2) * h)
                
                color = (0, 255, 0) # Green for all labels
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, class_names[int(cls_id)], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(os.path.join(output_dir, f"verify_{img_name}"), img)
        count += 1
        if count >= num_samples:
            break
    print(f"Verification images saved to {output_dir}")

if __name__ == "__main__":
    verify_refined(
        image_dir="data/refined/val/images",
        label_dir="data/refined/val/labels",
        output_dir="debug_refined_viz",
        class_names={0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"}
    )
