from ultralytics import YOLO
import os

def train():
    # 1. Load the model
    # Use 'yolo11s.pt' (Small) as recommended for better performance than Nano but faster than Large
    # If starting from scratch, you can use 'yolo11s.yaml' to build a new model, but transfer learning is recommended.
    model = YOLO('yolo11s.pt') 

    # 2. Train the model
    # data='data.yaml' -> You must create this file pointing to your dataset
    # epochs=50 -> Adjustable
    # imgsz=640 -> Panorama images are wide, so you might want 1024 or higher if your GPU allows.
    # Note: If memory is an issue, reduce batch size.
    
    print("Starting training...")
    results = model.train(
        data='data.yaml', 
        epochs=100, 
        imgsz=1024, # Higher resolution for panoramas
        batch=4,    # Adjust based on GPU memory
        name='dentex_yolov11s',
        patience=20, # Early stopping
        save=True
    )
    
    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == '__main__':
    # Ensure data.yaml exists
    if not os.path.exists('data.yaml'):
        print("Error: 'data.yaml' file not found. Please create it first.")
        print("Example content:")
        print("path: ./data")
        print("train: images/train")
        print("val: images/val")
        print("names: {0: 'Caries'}")
    else:
        train()
