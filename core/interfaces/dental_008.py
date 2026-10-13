import os
import sys
import cv2
import torch
import numpy as np

# 파이썬 경로에 서브모듈 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, "../../../Dental_008/src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ultralytics import YOLO
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

try:
    from numbering.arch_sequence_matcher import assign_fdi_labels
    from numbering.fdi_corrector import correct_fdi_numbers
except ImportError:
    pass # Will handle gracefully if path issues exist

def init_008_model():
    """Dental_008 YOLOv8 치아 식별 모델을 초기화하여 반환합니다."""
    possible_paths = [
        os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/models/yolov8m_best.pt")),
        os.path.abspath(os.path.join(current_dir, "../../../Dental_008/weights/yolov8m_best.pt")),
        os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/models/yolov8m_best.onnx")),
        os.path.abspath(os.path.join(current_dir, "../../../Dental_008/weights/yolov8m_best.onnx")),
        os.path.abspath(os.path.join(current_dir, "../../../Dental_008/weights/yolov8m-seg.pt")),
    ]
    
    ckpt_path = None
    for p in possible_paths:
        if os.path.exists(p):
            ckpt_path = p
            break
            
    if ckpt_path is None:
        print("Warning: Dental_008 model weights not found.")
        return None
        
    try:
        print(f"Loading Dental_008 model from: {ckpt_path}")
        model = YOLO(ckpt_path)
    except Exception as e:
        print(f"Failed to load Dental_008 YOLO model: {e}")
        model = None
    
    return model

def run_tooth_segmentation(image: np.ndarray, model, device, conf_threshold=0.20, iou_threshold=0.50, criteria=None) -> dict:
    """
    YOLOv8 및 2-Stage Sequence Matcher를 사용하여 치아 식별 및 영역 분할을 수행합니다.
    임상 기준 SSOT(과잉치 SEPARATE_ENTITY 정책) 및 하류 호환성(contours 추출)을 지원합니다.
    """
    h, w, _ = image.shape
    
    # YOLO 추론 (Main Workstation RTX 5080 verified optimal: conf=0.20, iou=0.50)
    results = model(image, verbose=False, conf=conf_threshold, iou=iou_threshold)[0]
    
    pred_boxes = results.boxes.xyxy.to(device) if results.boxes else torch.zeros(0,4).to(device)
    pred_scores = results.boxes.conf.to(device) if results.boxes else torch.zeros(0).to(device)
    
    # Resize masks
    if results.masks is not None:
        pred_masks_resized = torch.nn.functional.interpolate(
            results.masks.data.float().unsqueeze(1), 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1).to(device)
    else:
        pred_masks_resized = torch.zeros((0, h, w)).to(device)
        
    # FDI Numbering (2-Stage) with Clinical Uncertainty Flagging
    uncertain_mask = torch.zeros(len(pred_boxes), dtype=torch.bool, device=device)
    try:
        pred_labels_fdi = assign_fdi_labels(pred_boxes, pred_scores, w, h)
        pred_labels_fdi, unc_flags = correct_fdi_numbers(pred_boxes, pred_labels_fdi, return_uncertainty=True)
        if hasattr(unc_flags, 'to'):
            uncertain_mask = unc_flags.to(device)
        else:
            uncertain_mask = torch.tensor(unc_flags, dtype=torch.bool, device=device)
    except Exception as e:
        print(f"Sequence Matcher fallback: {e}")
        pred_labels_fdi = torch.zeros(len(pred_boxes), dtype=torch.int64).to(device)
    
    # Supernumerary policy handling (SEPARATE_ENTITY)
    super_policy = "SEPARATE_ENTITY"
    if criteria and "supernumerary_policy" in criteria:
        super_policy = criteria["supernumerary_policy"]

    boxes_all = pred_boxes.cpu().numpy()
    labels_all = pred_labels_fdi.cpu().numpy()
    scores_all = pred_scores.cpu().numpy()
    masks_all = (pred_masks_resized.cpu().numpy() > 0.5)
    unc_all = uncertain_mask.cpu().numpy()

    # Identify unassigned but high-confidence teeth as potential supernumerary
    is_supernumerary = np.zeros(len(boxes_all), dtype=bool)
    if super_policy == "SEPARATE_ENTITY" and len(boxes_all) > 0:
        center_x = w / 2.0
        centers_y = (boxes_all[:, 1] + boxes_all[:, 3]) / 2.0
        sorted_y = np.sort(centers_y)
        if len(sorted_y) > 1:
            gaps = np.diff(sorted_y)
            max_gap_idx = np.argmax(gaps)
            split_y = (sorted_y[max_gap_idx] + sorted_y[max_gap_idx + 1]) / 2.0
        else:
            split_y = np.median(centers_y)
        
        for i in range(len(boxes_all)):
            if labels_all[i] == 0 and scores_all[i] >= conf_threshold:
                bx1, by1, bx2, by2 = boxes_all[i]
                bcx = (bx1 + bx2) / 2.0
                bcy = (by1 + by2) / 2.0
                is_upper = (bcy < split_y)
                is_left_img = (bcx <= center_x) # patient right (Q1/Q4)
                
                # Assign Quadrant + 9 (e.g. 19, 29, 39, 49)
                if is_upper:
                    quad_sn = 19 if is_left_img else 29
                else:
                    quad_sn = 49 if is_left_img else 39
                    
                labels_all[i] = quad_sn
                is_supernumerary[i] = True

    # Filter valid labels (> 0)
    valid_mask = labels_all > 0
    boxes_valid = boxes_all[valid_mask]
    masks_valid = masks_all[valid_mask]
    labels_valid = labels_all[valid_mask]
    scores_valid = scores_all[valid_mask]
    unc_valid = unc_all[valid_mask]
    super_valid = is_supernumerary[valid_mask]

    result = {
        'boxes': [],
        'masks': [],
        'contours': [],
        'fdi_labels': [],
        'scores': [],
        'uncertain': [],
        'is_supernumerary': []
    }

    for i in range(len(boxes_valid)):
        box = boxes_valid[i]
        mask = masks_valid[i]
        result['boxes'].append(box)
        result['masks'].append(mask)
        result['fdi_labels'].append(int(labels_valid[i]))
        result['scores'].append(scores_valid[i])
        result['uncertain'].append(bool(unc_valid[i]))
        result['is_supernumerary'].append(bool(super_valid[i]))
        
        # Extract contour from mask for downstream modules
        mask_uint8 = (mask.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts and len(cnts) > 0:
            largest_cnt = max(cnts, key=cv2.contourArea)
            result['contours'].append(largest_cnt)
        else:
            x1, y1, x2, y2 = box
            bbox_cnt = np.array([[[int(x1), int(y1)]], [[int(x2), int(y1)]], [[int(x2), int(y2)]], [[int(x1), int(y2)]]], dtype=np.int32)
            result['contours'].append(bbox_cnt)
        
    return result

from huggingface_hub import hf_hub_download

def is_lfs_pointer(path: str) -> bool:
    """Check if file is a Git LFS pointer instead of a real binary checkpoint"""
    if not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) < 1024:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                return 'git-lfs' in first_line or first_line.startswith('version https://git-lfs')
    except Exception:
        pass
    return False

def init_008_classifier():
    """Dental_008 유치 이진 분류기를 안전하게 초기화하여 반환합니다."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    ckpt_path = os.path.abspath(os.path.join(current_dir, "../../../Dental_008/weights/pretrained/classifier_best.pth"))
    if not os.path.exists(ckpt_path) or is_lfs_pointer(ckpt_path):
        try:
            print("Local file is absent or Git LFS pointer. Downloading deciduous classifier from Hugging Face...")
            ckpt_path = hf_hub_download(repo_id="chemahc94/dentex-tooth-segmentation", filename="classifier_best.pth")
        except Exception as e:
            print(f"Failed to download classifier from Hugging Face: {e}")
            return None
            
    if os.path.exists(ckpt_path) and not is_lfs_pointer(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            model.eval()
            return model
        except Exception as e:
            print(f"Warning: Failed to load classifier weights ({e}). Gracefully falling back.")
            return None
    
    return None

def run_deciduous_classification(image: np.ndarray, model, device) -> bool:
    """
    유치 존재 여부를 분류합니다.
    """
    if model is None:
        return False
        
    try:
        img = Image.fromarray(image.astype('uint8')).convert('RGB')
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_t = val_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_t).squeeze(0)
            prob = torch.sigmoid(outputs)
            is_child = prob.item() > 0.5
            
        return is_child
    except Exception as e:
        print(f"Deciduous classification warning: {e}")
        return False
