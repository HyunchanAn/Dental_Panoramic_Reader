import urllib.request
import zipfile
import os

def download_and_extract():
    dataset_url = "https://zenodo.org/api/records/4457648/files/Panoramic%20radiography%20database.zip/content"
    data_dir = "data"
    zip_path = os.path.join(data_dir, "Panoramic_radiography_database.zip")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Downloading dataset from {dataset_url}...")
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Download complete. Extracting files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        print("Extraction complete.")
        # 옵션: 용량 확보를 위해 zip 파일 삭제
        os.remove(zip_path)
        print("Zip file cleaned up.")
    except Exception as e:
        print(f"Error during download or extraction: {e}")

if __name__ == "__main__":
    download_and_extract()
