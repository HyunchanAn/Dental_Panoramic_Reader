import os
import sys
import torch
import cv2
import numpy as np

# 파이썬 경로에 서브모듈 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_004/src"))
if module_path not in sys.path:
    sys.path.append(module_path)

from pano_clear.model import SwinIRLight
from pano_clear.preprocess import PanoPreprocessor
from pano_clear.tiling import PanoTiler
import yaml

def init_004_model():
    """Dental_004 SwinIR 모델을 초기화하여 반환합니다."""
    config_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_004/config/base_config.yaml"))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    )
    
    ckpt_path = os.path.abspath(os.path.join(current_dir, "../../modules/Dental_004/checkpoints/pano_swinir_epoch_100.pth"))
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    return model, config

def run_super_resolution(image: np.ndarray, model, config, device) -> np.ndarray:
    """초해상화 파이프라인 수행"""
    preprocessor = PanoPreprocessor()
    tiler = PanoTiler(tile_size=config['dataset']['patch_size'], overlap=32, upscale=config['model']['upscale'])
    
    # 1. 흑백 변환 및 정규화
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    # 2. 타일링 추론
    with torch.no_grad():
        tiles, positions = tiler.extract_tiles(img_tensor)
        out_tiles = []
        for tile in tiles:
            # 패딩 처리
            _, _, h, w = tile.shape
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), 'reflect')
                
            out_tile = model(tile)
            
            # 패딩 제거
            if pad_h > 0 or pad_w > 0:
                out_tile = out_tile[:, :, :h * config['model']['upscale'], :w * config['model']['upscale']]
                
            out_tiles.append(out_tile)
            
        out_tensor = tiler.merge_tiles(out_tiles, positions, (img_tensor.shape[2], img_tensor.shape[3]))
        
    out_img = out_tensor.squeeze().cpu().numpy()
    out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
    return out_img
