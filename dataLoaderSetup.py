import os
import torch
from torch.utils.data import Datset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class MeatFreshnessDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        root_dir: path to dataset
        Expected structure:
        root_dir/
          fresh/
            image1.jpg, image2.jpg, ...
          half_fresh/
          spoiled/
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        self.classes = ['fresh', 'half_fresh', 'spoiled']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                      self.samples.append((
                            os.path.join(class_dir, img_file),
                            self.class_to_idx[class_name]
                        ))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

  def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform
