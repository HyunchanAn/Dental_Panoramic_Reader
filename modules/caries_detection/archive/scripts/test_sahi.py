from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction
import os
import cv2

def run_sahi_inference(model_path, image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    detection_model = UltralyticsDetectionModel(
        model_path=model_path,
        confidence_threshold=0.25,
        device="cuda:0",
    )
    
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    
    print(f"Detected {len(result.object_prediction_list)} objects using SAHI.")
    
    # Manual visualization since export method name might have changed
    img = cv2.imread(image_path)
    for pred in result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(img, f"{pred.category.name}: {pred.score.value:.2f}", (int(bbox[0]), int(bbox[1])-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    out_path = os.path.join(output_dir, f"sahi_viz_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, img)
    print(f"Manual visualization saved to: {out_path}")

if __name__ == "__main__":
    MODEL_PATH = "models/best.pt"
    IMAGE_PATH = "data/processed/val/images/val_0.png" 
    OUT_DIR = "sahi_results"
    
    if os.path.exists(MODEL_PATH) and os.path.exists(IMAGE_PATH):
        run_sahi_inference(MODEL_PATH, IMAGE_PATH, OUT_DIR)
