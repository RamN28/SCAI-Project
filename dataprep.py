import torch
import pandas as pd
from torch.utils.data import Dataset

class MeatDataset(Dataset):
    def __init__(self, image_paths, environmental_data, labels):
        self.image_paths = image_paths
        self.env_data = environmental_data  # [temperature, time_since_slaughter, location_data]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # You must define this function somewhere
        image = load_and_preprocess_image(self.image_paths[idx])

        # Convert environmental features to a tensor
        env_features = torch.tensor(self.env_data[idx], dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, env_features, label



