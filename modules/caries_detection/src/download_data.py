import kagglehub
import shutil
import os
import argparse

def download_kaggle_dataset(target_dir):
    print("Downloading dataset from Kaggle...")
    # Download latest version using the snippet provided by user
    path = kagglehub.dataset_download("truthisneverlinear/childrens-dental-panoramic-radiographs-dataset")
    print(f"Dataset downloaded to cache at: {path}")

    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created target directory: {target_dir}")

    # Copy files to target
    print(f"Copying files to {target_dir}...")
    
    # Iterate over the files in the download path and move/copy them
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if os.path.exists(d):
               print(f"Removing existing directory: {d}")
               shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            if os.path.exists(d):
                print(f"Overwriting existing file: {d}")
            shutil.copy2(s, d)
            
    print("Download and copy complete!")
    print(f"Files are now available in: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kaggle dataset")
    parser.add_argument("--dir", type=str, default=None, help="Target directory for download")
    args = parser.parse_args()

    if args.dir:
        DOWNLOAD_DIR = args.dir
    else:
        # Default to PROJECT_ROOT/data
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
        DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "data")
    
    download_kaggle_dataset(DOWNLOAD_DIR)
