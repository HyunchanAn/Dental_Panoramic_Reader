import cv2
import os
import glob
from pathlib import Path
from tqdm import tqdm

def draw_yolo_labels(image_path, label_path, output_path, class_names):
    """
    Draws YOLO labels on an image and saves the result.
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    h, w, _ = img.shape
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, nw, nh = map(float, parts[1:])
                    
                    # Convert YOLO to pixel coordinates
                    x1 = int((cx - nw / 2) * w)
                    y1 = int((cy - nh / 2) * h)
                    x2 = int((cx + nw / 2) * w)
                    y2 = int((cy + nh / 2) * h)
                    
                    # Draw box
                    color = (0, 255, 0) # Green
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    label = class_names.get(cls_id, str(cls_id))
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imwrite(output_path, img)

def bulk_visualize(image_dir, label_dir, output_dir, class_names, limit=20):
    """
    Visualizes a batch of images and their labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    
    count = 0
    for img_path in tqdm(image_files[:limit], desc="Visualizing labels"):
        basename = os.path.basename(img_path)
        label_file = os.path.splitext(basename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)
        
        output_path = os.path.join(output_dir, f"viz_{basename}")
        draw_yolo_labels(img_path, label_path, output_path, class_names)
        count += 1
    
    print(f"Visualization complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    # Settings
    IMG_DIR = "data/processed/val/images"
    LBL_DIR = "data/processed/val/labels"
    OUT_DIR = "debug_viz"
    
    CLASSES = {
        0: "Impacted",
        1: "Caries",
        2: "Periapical",
        3: "Deep Caries"
    }
    
    if os.path.exists(IMG_DIR):
        bulk_visualize(IMG_DIR, LBL_DIR, OUT_DIR, CLASSES)
    else:
        print(f"Directory not found: {IMG_DIR}")
