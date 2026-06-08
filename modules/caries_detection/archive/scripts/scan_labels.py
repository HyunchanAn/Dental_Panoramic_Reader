import os
import json
from collections import Counter
import sys

def scan_labels(root_dir, outfile):
    outfile.write(f"Scanning {root_dir}...\n")
    unique_labels = Counter()
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "shapes" in data:
                            for shape in data["shapes"]:
                                label = shape.get("label", "unknown")
                                unique_labels[label] += 1
                except Exception as e:
                    outfile.write(f"Error reading {path}: {e}\n")
                    
    outfile.write("Unique Labels Found:\n")
    for label, count in unique_labels.most_common():
        outfile.write(f"{label}: {count}\n")
    outfile.write("-" * 20 + "\n")

if __name__ == "__main__":
    with open("labels_report.txt", "w", encoding="utf-8") as f:
        # Scan the Pediatric dataset
        path1 = r"c:\Users\chema\Github\Caries_Detection_from_Panoramic\data\Dental_dataset\Pediatric dental disease detection dataset"
        scan_labels(path1, f)

        # Check the other dataset
        path2 = r"c:\Users\chema\Github\Caries_Detection_from_Panoramic\data\Dental_dataset\Childrens dental caries segmentation dataset"
        scan_labels(path2, f)
        
        # Check the wanmugui dataset
        path3 = r"c:\Users\chema\Github\Caries_Detection_from_Panoramic\data\raw\childrens-dental-panoramic-x-ray-dataset"
        scan_labels(path3, f)
