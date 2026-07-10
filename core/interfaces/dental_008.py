import os
import sys
import torch
import numpy as np

# 파이썬 경로에 서브모듈 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from dentex_seg.model import get_instance_segmentation_model
from dentex_seg.dataset import get_fdi_to_class_id
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

def init_008_model():
    """Dental_008 Mask R-CNN 모델을 초기화하여 반환합니다."""
    # 모델 로드 (53개 클래스: 영구치 32 + 유치 20 + 배경 1)
    num_classes = 53
    model = get_instance_segmentation_model(num_classes)
    
    ckpt_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/weights/mask_rcnn_dentex_best.pth"))
    # best.pth가 없으면 fallback
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/weights/mask_rcnn_dentex_epoch_5.pth"))
        
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def run_tooth_segmentation(image: np.ndarray, model, device, conf_threshold=0.5) -> dict:
    """
    치아 식별 및 영역 분할을 수행합니다.
    Args:
        image: RGB numpy array (H, W, 3)
    Returns:
        dict: {'boxes': [...], 'masks': [...], 'fdi_labels': [...], 'scores': [...]}
    """
    _, id_to_fdi = get_fdi_to_class_id()
    
    # 텐서 변환
    image_tensor = torch.as_tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
        
    pred = predictions[0]
    masks = (pred['masks'] > 0.5).squeeze(1).cpu().numpy()
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    result = {
        'boxes': [],
        'masks': [],
        'fdi_labels': [],
        'scores': []
    }
    
    for i in range(len(boxes)):
        if scores[i] < conf_threshold:
            continue
            
        fdi_number = id_to_fdi.get(labels[i], "Unknown")
        
        result['boxes'].append(boxes[i])
        result['masks'].append(masks[i])
        result['fdi_labels'].append(fdi_number)
        result['scores'].append(scores[i])
        
    return result

def init_008_classifier():
    """Dental_008 유치 이진 분류기를 초기화하여 반환합니다."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    ckpt_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_008/weights/pretrained/classifier_best.pth"))
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
