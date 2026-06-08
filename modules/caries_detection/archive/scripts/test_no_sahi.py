from ultralytics import YOLO
import os

model = YOLO("models/best.pt")
results = model.predict("data/processed/val/images/val_0.png", conf=0.25)
print(f"Detected {len(results[0].boxes)} objects WITHOUT SAHI.")
