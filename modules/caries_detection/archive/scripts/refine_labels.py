import os
import shutil
import pandas as pd
import cv2
import ast

def refine_labels(csv_path, base_dir, output_dir, conf_threshold=0.4):
    # Load FP candidates
    df = pd.read_csv(csv_path)
    # Filter by confidence
    df_refined = df[df["conf"] >= conf_threshold].copy()
    print(f"Filtering candidates with confidence >= {conf_threshold}: {len(df_refined)} cases found.")

    # Class map (reverse) - Sync with data.yaml
    CLASS_MAP = {"Impacted": 0, "Caries": 1, "Periapical Lesion": 2, "Deep Caries": 3, "Periapical": 2}

    # Create output directory structure
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for split in ["train", "val"]:
        src_img_dir = os.path.join(base_dir, split, "images")
        src_lbl_dir = os.path.join(base_dir, split, "labels")
        dst_img_dir = os.path.join(output_dir, split, "images")
        dst_lbl_dir = os.path.join(output_dir, split, "labels")
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        if not os.path.exists(src_img_dir):
            continue

        print(f"Refining {split} split...")
        
        # 1. Copy ALL images from source to destination
        for img_file in os.listdir(src_img_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))

        # 2. Copy ALL existing labels from source to destination
        for lbl_file in os.listdir(src_lbl_dir):
            if lbl_file.endswith('.txt'):
                shutil.copy(os.path.join(src_lbl_dir, lbl_file), os.path.join(dst_lbl_dir, lbl_file))

        # 3. Add refined labels
        split_df = df_refined[df_refined["split"] == split]
        for _, row in split_df.iterrows():
            img_name = row["image"]
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            dst_lbl_path = os.path.join(dst_lbl_dir, lbl_name)
            
            # Ensure file exists (even if it was a background image)
            if not os.path.exists(dst_lbl_path):
                open(dst_lbl_path, "w").close()

            # Get image size for normalization
            img_path = os.path.join(src_img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: 
                print(f"Warning: Could not read {img_path}")
                continue
            h, w = img.shape[:2]
            
            # box is [x1, y1, x2, y2]
            box = ast.literal_eval(row["box"])
            x1, y1, x2, y2 = box
            
            # Convert to YOLO format
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            nw = (x2 - x1) / w
            nh = (y2 - y1) / h
            
            cls_id = CLASS_MAP.get(row["class"], 1) # Default to Caries if not found
            
            # Append to file with careful newline handling
            with open(dst_lbl_path, "r") as f:
                content = f.read()
            
            # If content doesn't end with newline and is not empty, add one
            with open(dst_lbl_path, "a") as f:
                if content and not content.endswith('\n'):
                    f.write('\n')
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"Label refinement complete. Refined data saved to {output_dir}")

if __name__ == "__main__":
    refine_labels(
        csv_path="fp_candidates.csv",
        base_dir="c:/Users/chema/Github/Caries_Detection_from_Panoramic/data/processed",
        output_dir="c:/Users/chema/Github/Caries_Detection_from_Panoramic/data/refined",
        conf_threshold=0.4
    )
