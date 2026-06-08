import os

def count_labels(label_dir, class_names):
    counts = {k: 0 for k in class_names}
    if not os.path.exists(label_dir): return counts
    for lbl in os.listdir(label_dir):
        if not lbl.endswith(".txt"): continue
        for line in open(os.path.join(label_dir, lbl)):
            parts = line.strip().split()
            if len(parts) < 1: continue
            try:
                cid = int(parts[0])
                if cid in counts: counts[cid] += 1
            except: continue
    return counts

if __name__ == "__main__":
    CLASSES = {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"}
    print("Original Val Counts:")
    print(count_labels("c:/Users/chema/Github/Caries_Detection_from_Panoramic/data/processed/val/labels", CLASSES))
    print("\nRefined Val Counts:")
    print(count_labels("c:/Users/chema/Github/Caries_Detection_from_Panoramic/data/refined/val/labels", CLASSES))
    
    print("\nOriginal Train Counts:")
    print(count_labels("c:/Users/chema/Github/Caries_Detection_from_Panoramic/data/processed/train/labels", CLASSES))
    print("\nRefined Train Counts:")
    print(count_labels("c:/Users/chema/Github/Caries_Detection_from_Panoramic/data/refined/train/labels", CLASSES))
