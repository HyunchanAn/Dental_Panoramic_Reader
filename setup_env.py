import os
import urllib.request
from huggingface_hub import hf_hub_download

# Mapping of required models and their HuggingFace / Direct paths
MODELS = {
    "Dental_001": {
        "cvm_model": ("live-track/dental-cvm", "best_cvm_model.pth"),
        "yolo_model": ("live-track/dental-yolo", "yolov8m_custom.pt"),
        "unet_model": ("live-track/dental-unet", "resnet50_unet.pth")
    },
    "Dental_002": {
        "caries_model": ("live-track/dental-caries", "best_refined.pt")
    },
    "Dental_003": {
        "boneloss_model": ("live-track/dental-boneloss", "best.pt"),
        "classifier_model": ("live-track/dental-pano-classifier", "pano_classifier.pt")
    },
    "Dental_004": {
        "sr_model": ("live-track/dental-sr", "best_swinir.pth")
    },
    "Dental_008": {
        "seg_model": ("live-track/dental-seg", "yolov8m-seg.pt")
    },
    "Dental_011": {
        "age_model": ("live-track/dental-age", "best_hybrid_age_model.pth")
    },
    "Dental_012": {
        "periapical_model": ("live-track/dental-periapical", "best.pt")
    },
    "Dental_013": {
        "restoration_model": ("live-track/dental-restoration", "best.pt")
    }
}

def get_model_dir(module_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "modules", module_name, "models")

def download_models():
    print("Starting automated download of all required weights...")
    for module, weights in MODELS.items():
        dest_dir = get_model_dir(module)
        os.makedirs(dest_dir, exist_ok=True)
        for name, (repo, filename) in weights.items():
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(dest_path):
                print(f"[{module}] {filename} already exists. Skipping.")
                continue
            print(f"[{module}] Downloading {filename} from HuggingFace ({repo})...")
            try:
                hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)
                print(f"  -> Successfully downloaded to {dest_path}")
            except Exception as e:
                raise RuntimeError(f"Critical Error: Failed to download {filename} from {repo}. "
                                   f"Please check your network connection or HuggingFace token. "
                                   f"Detailed error: {e}")

if __name__ == "__main__":
    download_models()
    print("All downloads complete. The environment is ready.")
