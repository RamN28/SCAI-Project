# SCAI Meat Freshness Classification Project

This project uses PyTorch to classify pork meat into three categories:
- Fresh
- Half-Fresh  
- Spoiled
using kaggle dataset: https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset/data

## Project Structure
- `CNNModel.py` - Simple CNN model for image classification
- `dataLoaderSetup.py` - Data loading and preprocessing
- `train.py` - Main training script
- `requirements.txt` - Project dependencies

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Organize your dataset in `data/raw/meat_freshness/` with subfolders: fresh/, half_fresh/, spoiled/
3. Run training: `python train.py`

## Dataset
We're using the Meat Freshness dataset from Kaggle to train our model.
