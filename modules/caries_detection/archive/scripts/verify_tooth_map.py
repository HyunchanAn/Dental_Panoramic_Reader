import cv2
import numpy as np
import os
import sys

# Add current directory to path so it can find tooth_mapper
sys.path.append(os.path.dirname(__file__))
from tooth_mapper import assign_tooth_number

def visualize_tooth_map(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    h, w, _ = img.shape
    
    overlay = img.copy()
    cv2.line(overlay, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    cv2.line(overlay, (0, h//2), (w, h//2), (255, 255, 255), 2)
    
    step_x = w // 16
    step_y = h // 8
    
    for y in range(step_y//2, h, step_y):
        for x in range(step_x//2, w, step_x):
            tooth_num = assign_tooth_number(x, y, w, h)
            quad = tooth_num // 10
            colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)}
            color = colors.get(quad, (255, 255, 255))
            cv2.circle(overlay, (x, y), 5, color, -1)
            cv2.putText(overlay, f"#{tooth_num}", (x-20, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    result = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    cv2.imwrite(output_path, result)
    print(f"Tooth mapping visualization saved to: {output_path}")

if __name__ == "__main__":
    TEST_IMG = "data/processed/val/images/val_0.png"
    OUT_PATH = "debug_tooth_map.png"
    if os.path.exists(TEST_IMG):
        visualize_tooth_map(TEST_IMG, OUT_PATH)
    else:
        print(f"Test image not found: {TEST_IMG}")
