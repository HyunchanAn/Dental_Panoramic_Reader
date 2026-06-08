import numpy as np

def crop_roi(image_rgb: np.ndarray, bbox: list, padding: int = 20) -> tuple:
    """
    원본 이미지에서 BBox 기반으로 치아 주변부(ROI)를 크롭합니다.
    
    Args:
        image_rgb: 원본 RGB 이미지
        bbox: [cx, cy, w, h, angle] 형식의 바운딩 박스
        padding: 크롭 시 추가할 여백 (픽셀)
        
    Returns:
        (cropped_image, offset_x, offset_y): 크롭된 이미지와 원본 이미지 대비 좌상단 오프셋 좌표
    """
    cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    
    img_h, img_w = image_rgb.shape[:2]
    
    x_min = max(0, int(cx - w / 2) - padding)
    y_min = max(0, int(cy - h / 2) - padding)
    x_max = min(img_w, int(cx + w / 2) + padding)
    y_max = min(img_h, int(cy + h / 2) + padding)
    
    cropped_image = image_rgb[y_min:y_max, x_min:x_max]
    
    return cropped_image, x_min, y_min
