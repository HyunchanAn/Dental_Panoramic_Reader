import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def export_classifier(pt_path: str = None):
    device = torch.device('cpu')
    pt_path = pt_path or os.getenv("PANO_CLASSIFIER_PT", "modules/Dental_003/models/pano_classifier.pt")
    onnx_path = "modules/Dental_003/models/pano_classifier.onnx"
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    print(f"Loading MobileNetV3 weights from {pt_path}")
    model = mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    export_classifier()
