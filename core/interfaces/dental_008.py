import os
import sys
import torch
import numpy as np

# 파이썬 경로에 서브모듈 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from ultralytics import YOLO
import torch.nn as nn
from PIL import Image

try:
    from numbering.arch_sequence_matcher import assign_fdi_labels
    from numbering.fdi_corrector import correct_fdi_numbers
except ImportError:
    pass # Will handle gracefully if path issues exist

def init_008_model():
    """Dental_008 YOLOv8 모델을 초기화하여 반환합니다."""
    ckpt_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/yolov8m-seg.pt"))
    if not os.path.exists(ckpt_path):
        # Fallback to weights directory
        ckpt_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/weights/yolov8m-seg.pt"))
        
    try:
        model = YOLO(ckpt_path)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        model = None
    
    return model

def run_tooth_segmentation(image: np.ndarray, model, device, conf_threshold=0.5) -> dict:
    """
    YOLOv8 및 2-Stage Sequence Matcher를 사용하여 치아 식별 및 영역 분할을 수행합니다.
    Args:
        image: RGB numpy array (H, W, 3)
    Returns:
        dict: {'boxes': [...], 'masks': [...], 'fdi_labels': [...], 'scores': [...]}
    """
    h, w, _ = image.shape
    
    # YOLO 추론
    results = model(image, verbose=False, conf=conf_threshold, iou=0.4)[0]
    
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
        
    # FDI Numbering (2-Stage)
    pred_labels_fdi = assign_fdi_labels(pred_boxes, pred_scores, w, h)
    pred_labels_fdi = correct_fdi_numbers(pred_boxes, pred_labels_fdi)
    
    # Filter valid labels (> 0)
    valid_mask = pred_labels_fdi > 0
    pred_boxes = pred_boxes[valid_mask]
    pred_masks_resized = pred_masks_resized[valid_mask]
    pred_labels_fdi = pred_labels_fdi[valid_mask]
    pred_scores = pred_scores[valid_mask]
    
    boxes_np = pred_boxes.cpu().numpy()
    masks_np = (pred_masks_resized.cpu().numpy() > 0.5)
    fdi_np = pred_labels_fdi.cpu().numpy()
    scores_np = pred_scores.cpu().numpy()
    
    result = {
        'boxes': [],
        'masks': [],
        'fdi_labels': [],
        'scores': []
    }
    
    for i in range(len(boxes_np)):
        result['boxes'].append(boxes_np[i])
        result['masks'].append(masks_np[i])
        result['fdi_labels'].append(int(fdi_np[i]))
        result['scores'].append(scores_np[i])
        
    return result

from huggingface_hub import hf_hub_download

def init_008_classifier():
    """Dental_008 유치 이진 분류기를 초기화하여 반환합니다."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    ckpt_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/weights/pretrained/classifier_best.pth"))
    if not os.path.exists(ckpt_path):
        try:
            print("Downloading deciduous classifier from Hugging Face...")
            ckpt_path = hf_hub_download(repo_id="chemahc94/dentex-tooth-segmentation", filename="classifier_best.pth")
        except Exception as e:
            print(f"Failed to download classifier from Hugging Face: {e}")
            
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def run_deciduous_classification(image: np.ndarray, model, device) -> bool:
    """
    유치 존재 여부를 분류합니다.
    Args:
        image: RGB numpy array (H, W, 3)
    Returns:
        bool: True if deciduous (Child) tooth is detected, False otherwise.
    """
    img = Image.fromarray(image.astype('uint8')).convert('RGB')
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_t = val_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t).squeeze(0) # shape (1,)
        # 이진 분류 (0: Adult, 1: Child) - sigmoid threshold 0.5
        prob = torch.sigmoid(outputs)
        is_child = prob.item() > 0.5
        
    return is_child
