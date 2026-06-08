from ultralytics import YOLO
import json
import os
from datetime import datetime

def run_evaluation(model_path, data_config, split='val'):
    """
    Evaluates the model on a specified dataset split and saves the metrics.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None

    print(f"Starting evaluation of {model_path} on {split} set...")
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data=data_config, split=split)
    
    # Extract metrics
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "data_config": data_config,
        "split": split,
        "metrics": {
            "mAP50": float(results.results_dict['metrics/mAP50(B)']),
            "mAP50-95": float(results.results_dict['metrics/mAP50-95(B)']),
            "precision": float(results.results_dict['metrics/precision(B)']),
            "recall": float(results.results_dict['metrics/recall(B)']),
            "fitness": float(results.fitness)
        }
    }
    
    # Print results summary
    print("\n" + "="*30)
    print("Evaluation Results Summary")
    print("="*30)
    for key, val in metrics["metrics"].items():
        print(f"{key:15}: {val:.4f}")
    print("="*30)
    
    # Save to file
    report_path = f"evaluation_report_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Full report saved to: {report_path}")
    return metrics

if __name__ == "__main__":
    MODEL_PATH = "models/best.pt"
    DATA_CONFIG = "data.yaml"
    
    if os.path.exists(DATA_CONFIG):
        run_evaluation(MODEL_PATH, DATA_CONFIG, split='val')
    else:
        print(f"Error: {DATA_CONFIG} not found.")
