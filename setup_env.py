import os
import shutil
import hashlib
import yaml
from huggingface_hub import hf_hub_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "weights_manifest.yaml")


def load_weights_manifest(manifest_path: str = MANIFEST_PATH) -> dict:
    """Load the single weights manifest YAML."""
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Weights manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("weights", {})


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_weight_file(weight_key: str, download_if_missing: bool = True) -> str:
    """Resolve local weight path from weights_manifest.yaml and optionally download from HuggingFace."""
    manifest = load_weights_manifest()
    if weight_key not in manifest:
        raise KeyError(f"Unknown weight key '{weight_key}' in weights_manifest.yaml")

    entry = manifest[weight_key]
    local_rel = entry["local_path"]
    dest_path = os.path.join(BASE_DIR, local_rel)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        expected_sha = entry.get("sha256")
        if expected_sha and compute_sha256(dest_path).lower() != expected_sha.lower():
            raise RuntimeError(f"SHA-256 mismatch for {dest_path}")
        return dest_path

    if not download_if_missing:
        raise FileNotFoundError(f"Weight file not found locally: {dest_path}")

    repo_id = entry["repo_id"]
    hf_filename = entry["hf_filename"]
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN") or None

    try:
        cached_path = hf_hub_download(repo_id=repo_id, filename=hf_filename, token=hf_token)
        shutil.copy2(cached_path, dest_path)
    except Exception as e:
        raise RuntimeError(
            f"Critical Error: Failed to download '{hf_filename}' from '{repo_id}' to '{dest_path}': {e}"
        ) from e

    expected_sha = entry.get("sha256")
    if expected_sha and compute_sha256(dest_path).lower() != expected_sha.lower():
        raise RuntimeError(f"SHA-256 verification failed for downloaded file: {dest_path}")

    return dest_path


def download_models():
    print("Starting automated download of all required weights from weights_manifest.yaml...")
    manifest = load_weights_manifest()
    for key, entry in manifest.items():
        local_rel = entry["local_path"]
        dest_path = os.path.join(BASE_DIR, local_rel)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
            print(f"[{key}] {local_rel} already exists. Skipping.")
            continue
        print(f"[{key}] Downloading {entry['hf_filename']} from {entry['repo_id']} -> {local_rel}...")
        ensure_weight_file(key, download_if_missing=True)
        print(f"  -> Successfully saved to {dest_path}")


if __name__ == "__main__":
    download_models()
    print("All downloads complete. The environment is ready.")
