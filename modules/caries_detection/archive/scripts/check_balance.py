import os
import glob
from collections import Counter
import pandas as pd

def count_classes(label_dir):
    """
    Counts the number of instances for each class in a YOLO label directory.
    """
    class_counts = Counter()
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1
    
    return class_counts, len(label_files)

def report_balance(data_root, class_names):
    """
    Analyzes and reports class distribution across train and val sets.
    """
    sets = ['train', 'val']
    report_data = []
    
    for s in sets:
        label_dir = os.path.join(data_root, s, "labels")
        if not os.path.exists(label_dir):
            print(f"Warning: {label_dir} not found.")
            continue
            
        counts, num_images = count_classes(label_dir)
        total_instances = sum(counts.values())
        
        print(f"\n--- {s.upper()} Set Statistics ---")
        print(f"Total Images: {num_images}")
        print(f"Total Instances: {total_instances}")
        
        for cls_id, name in class_names.items():
            count = counts.get(cls_id, 0)
            percentage = (count / total_instances * 100) if total_instances > 0 else 0
            avg_per_img = count / num_images if num_images > 0 else 0
            print(f"Class {cls_id} ({name:18}): {count:5} ({percentage:6.2f}%) [Avg: {avg_per_img:.2f}]")
            
            report_data.append({
                "Set": s,
                "Class_ID": cls_id,
                "Class_Name": name,
                "Count": count,
                "Percentage": percentage
            })
            
    # Save as CSV for further analysis if needed
    df = pd.DataFrame(report_data)
    df.to_csv("class_distribution_report.csv", index=False)
    print(f"\nFull report saved to class_distribution_report.csv")

if __name__ == "__main__":
    DATA_ROOT = "data/processed"
    CLASSES = {
        0: "Impacted",
        1: "Caries",
        2: "Periapical Lesion",
        3: "Deep Caries"
    }
    
    report_balance(DATA_ROOT, CLASSES)
