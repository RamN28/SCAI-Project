def download_dataset():
  if not os.path.exists('data/meat_freshness'):
        # Download from URL
        gdown.download('https://drive.google.com/uc?id=YOUR_FILE_ID', 'data/dataset.zip')
        # Extract and process
