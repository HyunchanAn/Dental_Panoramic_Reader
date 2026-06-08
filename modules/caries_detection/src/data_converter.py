import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def coco_to_yolo_bbox(bbox, img_w, img_h):
    """
    Converts COCO format [x_min, y_min, w, h] to YOLO format [center_x, center_y, norm_w, norm_h]
    with clipping to [0, 1].
    """
    x_min, y_min, w, h = bbox
    
    # Calculate centers and normalized dimensions
    center_x = (x_min + w / 2) / img_w
    center_y = (y_min + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    # Clipping to [0, 1] to handle edge cases
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))
    
    return center_x, center_y, norm_w, norm_h

def convert_dentex_to_yolo(json_path, image_dir, output_dir):
    """
    Converts DENTEX COCO-style JSON annotations to YOLO format.
    Uses 'categories_3' for disease classification.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create directories
    labels_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    # DENTEX uses categories_3 for diseases
    categories = {cat['id']: cat['name'] for cat in data['categories_3']}
    print(f"Categories found in {os.path.basename(json_path)}:")
    
    # YOLO ID mapping: Ensure 0-indexed sequential IDs
    sorted_cat_ids = sorted(categories.keys())
    cat_id_to_yolo_id = {cat_id: i for i, cat_id in enumerate(sorted_cat_ids)}
    
    for cat_id in sorted_cat_ids:
        print(f"Original ID: {cat_id} -> YOLO ID: {cat_id_to_yolo_id[cat_id]}, Name: {categories[cat_id]}")

    # Process images
    images_info = {img['id']: img for img in data['images']}
    
    converted_count = 0
    for img_id, img_info in tqdm(images_info.items(), desc=f"Converting {os.path.basename(json_path)}"):
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        # Source image path - handle potential subdirectories in file_name
        src_img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(src_img_path):
            # Try without subdirectory if file_name has one
            if '/' in file_name or '\\' in file_name:
                src_img_path = os.path.join(image_dir, os.path.basename(file_name))
            
            if not os.path.exists(src_img_path):
                continue 

        # Create label file
        label_file = os.path.splitext(os.path.basename(file_name))[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # In DENTEX, annotations are in 'annotations' list
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        if not annotations:
            continue
            
        yolo_lines = []
        for ann in annotations:
            # Check for category_id_3 (DENTEX disease task) or category_id
            cat_id = ann.get('category_id_3', ann.get('category_id'))
            
            if cat_id is None or cat_id not in cat_id_to_yolo_id:
                continue
                
            yolo_cls = cat_id_to_yolo_id[cat_id]
            bbox = ann['bbox'] # [x_min, y_min, width, height]
            
            center_x, center_y, norm_w, norm_h = coco_to_yolo_bbox(bbox, img_w, img_h)
            
            yolo_lines.append(f"{yolo_cls} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        if yolo_lines:
            with open(label_path, 'w') as f_out:
                f_out.write('\n'.join(yolo_lines))
            
            # Copy image to YOLO directory (use basename to avoid folder structure issues in YOLO)
            dst_img_path = os.path.join(images_output_dir, os.path.basename(file_name))
            shutil.copy2(src_img_path, dst_img_path)
            converted_count += 1
            
    print(f"Successfully converted {converted_count} images for {os.path.basename(json_path)}")

if __name__ == "__main__":
    # Base paths
    BASE_RAW = "data/raw"
    BASE_PROCESSED = "data/processed"
    
    # Train Data (Quadrant-Enumeration-Disease task)
    TRAIN_JSON = os.path.join(BASE_RAW, "training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json")
    TRAIN_IMG = os.path.join(BASE_RAW, "training_data/quadrant-enumeration-disease/xrays")
    TRAIN_OUT = os.path.join(BASE_PROCESSED, "train")
    
    # Val Data (Using validation_triple which contains disease labels)
    VAL_JSON = os.path.join(BASE_RAW, "DENTEX/DENTEX/validation_triple.json")
    VAL_IMG = os.path.join(BASE_RAW, "validation_test/validation_data/quadrant_enumeration_disease/xrays")
    VAL_OUT = os.path.join(BASE_PROCESSED, "val")
    
    # Run conversion for Train
    if os.path.exists(TRAIN_JSON):
        print(f"Converting Training Data from {TRAIN_JSON}...")
        convert_dentex_to_yolo(TRAIN_JSON, TRAIN_IMG, TRAIN_OUT)
    else:
        print(f"Train JSON not found: {TRAIN_JSON}")

    # Run conversion for Val
    if os.path.exists(VAL_JSON):
        print(f"\nConverting Validation Data from {VAL_JSON}...")
        convert_dentex_to_yolo(VAL_JSON, VAL_IMG, VAL_OUT)
    else:
        print(f"Val JSON not found: {VAL_JSON}")

