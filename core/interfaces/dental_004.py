import os
import sys
import torch
import cv2
import numpy as np

# 파이썬 경로에 내부 서브모듈 우선 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
module_root = os.path.join(repo_root, "modules", "Dental_004")
if not os.path.isdir(os.path.join(module_root, "src")):
    alt_root = os.path.join(os.path.dirname(repo_root), "Dental_004")
    if os.path.isdir(os.path.join(alt_root, "src")):
        module_root = alt_root
module_path = os.path.join(module_root, "src")
if module_path not in sys.path:
    sys.path.append(module_path)

from pano_clear.model import SwinIRLight
from pano_clear.preprocess import PanoPreprocessor
from pano_clear.tiling import PanoTiler
import yaml

def init_004_model():
    """Dental_004 SwinIR 모델을 초기화하여 반환합니다."""
    config_path = os.path.join(module_root, "config", "base_config.yaml")
    if not os.path.exists(config_path):
        print(f"Warning: Dental_004 config not found at {config_path}.")
        return None, None
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    model = SwinIRLight(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        embed_dim=config['model']['embed_dim'],
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        window_size=config['model']['window_size']
    )
    
    ckpt_candidates = [
        os.path.join(module_root, "models", "best_swinir.pth"),
        os.path.join(module_root, "checkpoints", "pano_swinir_epoch_100.pth"),
    ]
    ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if ckpt_path is None:
        print("Warning: Dental_004 checkpoint not found. Refusing to return uninitialized random-weight model.")
        return None, None

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return model, config
    except Exception as e:
        print(f"Failed to load Dental_004 checkpoint ({e}). Returning None.")
        return None, None

def run_super_resolution(image: np.ndarray, model, config, device) -> np.ndarray:
    """Run SwinIR super-resolution tiling pipeline (preserves input when model is unavailable)."""
    if model is None or config is None:
        return image
    upscale = int(config['model']['upscale'])
    tile_size = int(config['dataset']['patch_size'])
    overlap = min(32, max(8, tile_size // 4))
    tiler = PanoTiler(tile_size=tile_size, overlap=overlap, upscale=upscale)

    # 1. Grayscale conversion & normalization -> (C, H, W)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    orig_h, orig_w = gray.shape[:2]
    pad_h = max(0, tile_size - orig_h)
    pad_w = max(0, tile_size - orig_w)
    if pad_h > 0 or pad_w > 0:
        gray_padded = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    else:
        gray_padded = gray

    h_pad, w_pad = gray_padded.shape[:2]
    img_chw = torch.from_numpy(gray_padded).float().unsqueeze(0).to(device) / 255.0

    # 2. Tiled inference using PanoTiler.tile_image / merge_tiles
    with torch.no_grad():
        tiles, coords = tiler.tile_image(img_chw)
        out_tiles = []
        for idx in range(tiles.shape[0]):
            tile = tiles[idx : idx + 1]
            _, _, th, tw = tile.shape
            t_pad_h = (8 - th % 8) % 8
            t_pad_w = (8 - tw % 8) % 8
            if t_pad_h > 0 or t_pad_w > 0:
                tile = torch.nn.functional.pad(tile, (0, t_pad_w, 0, t_pad_h), 'reflect')
            out_tile = model(tile)
            if t_pad_h > 0 or t_pad_w > 0:
                out_tile = out_tile[:, :, : th * upscale, : tw * upscale]
            out_tiles.append(out_tile)

        stacked_out = torch.cat(out_tiles, dim=0)
        out_tensor = tiler.merge_tiles(stacked_out, coords, (1, h_pad * upscale, w_pad * upscale))

    out_gray = out_tensor.squeeze(0).cpu().numpy()
    out_gray = out_gray[: orig_h * upscale, : orig_w * upscale]
    out_gray = np.clip(out_gray * 255.0, 0, 255).astype(np.uint8)
    # Return 3-channel BGR so downstream Dental_008 (h, w, _ = image.shape) never crashes
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
