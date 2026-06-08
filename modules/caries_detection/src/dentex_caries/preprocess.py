import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.
    Works for both grayscale and color (RGB/BGR) images.
    """
    if len(image.shape) == 2:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    elif len(image.shape) == 3:
        # Color: Convert to LAB color space
        # L: Lightness, A/B: Color channels
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge back and convert to BGR
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return image

if __name__ == "__main__":
    # Test on a sample image
    import os
    test_img_path = "data/processed/val/images/val_0.png"
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        enhanced = apply_clahe(img)
        
        os.makedirs("debug_preprocess", exist_ok=True)
        cv2.imwrite("debug_preprocess/original.png", img)
        cv2.imwrite("debug_preprocess/enhanced_clahe.png", enhanced)
        print(f"CLAHE enhancement complete. Check 'debug_preprocess' folder.")
    else:
        print(f"Test image not found: {test_img_path}")
