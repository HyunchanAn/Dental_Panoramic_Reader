import os
import time
import torch
import cv2
import numpy as np

# 파이프라인 모듈 로드
from core.pipeline import PanoramicPipeline

def run_benchmark(image_path="test_images_PANO/sample1.png"):
    print("="*50)
    print("DENTEX E2E Pipeline Benchmark")
    print("="*50)
    
    if not os.path.exists(image_path):
        # Create dummy image if no sample exists
        print(f"Sample image {image_path} not found. Creating a dummy 2048x1024 image.")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        img = np.random.randint(0, 255, (1024, 2048, 3), dtype=np.uint8)
        cv2.imwrite(image_path, img)

    # 1. Initialization Time
    start_time = time.time()
    try:
        pipeline = PanoramicPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return
    init_time = time.time() - start_time
    print(f"[1] Pipeline Initialization Time: {init_time:.2f} seconds")

    # 2. VRAM Baseline
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        baseline_vram = torch.cuda.memory_allocated() / (1024**2)
        print(f"[2] Baseline VRAM Usage: {baseline_vram:.2f} MB")
    else:
        print("[2] CUDA not available. Running on CPU.")

    # 3. End-to-End Inference
    print("\nStarting E2E Inference...")
    start_time = time.time()
    result_json = pipeline.run(image_path)
    inference_time = time.time() - start_time
    
    print(f"\n[3] E2E Inference Time (Latency): {inference_time:.2f} seconds")

    # 4. VRAM Peak
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"[4] Peak VRAM Usage during inference: {peak_vram:.2f} MB")
    
    print("="*50)
    print("Benchmark Complete.")
    print(f"Modules executed: {list(result_json.get('modules', {}).keys())}")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
