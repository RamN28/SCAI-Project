import os
import zipfile

DATA_DIR = "data"
ZIP_FILE = "meat-freshness-image-dataset.zip"
KAGGLE_DATASET = "vinayakshanawad/meat-freshness-image-dataset"

def download_dataset():
    print("Downloading dataset from Kaggle...")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET}")

def extract_dataset():
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Check if data already exists (avoid redownloading)
    if len(os.listdir(DATA_DIR)) > 0:
        print("Dataset already exists. Skipping download.")
        return

    download_dataset()
    extract_dataset()

    print("Dataset downloaded and ready.")

if __name__ == "__main__":
    main()
