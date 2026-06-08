import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

class YOLOwrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # YOLO forward returns a list of tensors during training/val,
        # but for Grad-CAM we need the raw inference tensor.
        # Ultralytics models return a list/tuple. We take the first element.
        result = self.model(x)
        if isinstance(result, (list, tuple)):
            return result[0]
        return result

def get_xai_heatmap(model_path, image_path, target_layer_idx=21):
    """
    Generates an Eigen-CAM heatmap for a YOLOv11 model.
    Layer 21 is a final Neck layer before the Head, providing highly relevant features.
    """
    yolo = YOLO(model_path)
    pt_model = yolo.model
    wrapper = YOLOwrapper(pt_model)
    
    try:
        target_layers = [pt_model.model[target_layer_idx]]
    except IndexError:
        target_layers = [pt_model.model[-1]]
    
    cam = EigenCAM(model=wrapper, target_layers=target_layers)
    
    img = cv2.imread(image_path)
    if img is None: return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255
    
    h, w = img_rgb.shape[:2]
    new_h, new_w = (h // 32) * 32, (w // 32) * 32
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    with torch.set_grad_enabled(True):
        grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    grayscale_cam = cv2.resize(grayscale_cam, (w, h))
    
    # --- Edge-Suppression: Remove border artifacts ---
    margin_h, margin_w = int(h * 0.05), int(w * 0.05)
    mask = np.ones_like(grayscale_cam)
    mask[:margin_h, :] = 0; mask[-margin_h:, :] = 0
    mask[:, :margin_w] = 0; mask[:, -margin_w:] = 0
    # Smooth the mask edges
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    grayscale_cam = grayscale_cam * mask
    
    visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam

if __name__ == "__main__":
    MODEL = "models/best.pt"
    IMG = "data/processed/val/images/val_0.png"
    OUT = "debug_xai_heatmap.png"
    
    if os.path.exists(MODEL) and os.path.exists(IMG):
        viz, _ = get_xai_heatmap(MODEL, IMG)
        if viz is not None:
            cv2.imwrite(OUT, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
            print(f"XAI Heatmap saved to: {OUT}")
    else:
        print("Model or Image not found.")
