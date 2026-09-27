import os
import sys
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

def run_tooth_segmentation(image: np.ndarray, model, device, conf_threshold=0.20, iou_threshold=0.50) -> dict:
    """
    YOLOv8 및 2-Stage Sequence Matcher를 사용하여 치아 식별 및 영역 분할을 수행합니다.
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
    
    # Filter valid labels (> 0)
    valid_mask = pred_labels_fdi > 0
    pred_boxes = pred_boxes[valid_mask]
    pred_masks_resized = pred_masks_resized[valid_mask]
    pred_labels_fdi = pred_labels_fdi[valid_mask]
    pred_scores = pred_scores[valid_mask]
    uncertain_mask = uncertain_mask[valid_mask]
    
    boxes_np = pred_boxes.cpu().numpy()
    masks_np = (pred_masks_resized.cpu().numpy() > 0.5)
    fdi_np = pred_labels_fdi.cpu().numpy()
    scores_np = pred_scores.cpu().numpy()
    unc_np = uncertain_mask.cpu().numpy()
    
    result = {
        'boxes': [],
        'masks': [],
        'fdi_labels': [],
        'scores': [],
        'uncertain': []
    }
    
    for i in range(len(boxes_np)):
        result['boxes'].append(boxes_np[i])
        result['masks'].append(masks_np[i])
        result['fdi_labels'].append(int(fdi_np[i]))
        result['scores'].append(scores_np[i])
        result['uncertain'].append(bool(unc_np[i]))
        
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
