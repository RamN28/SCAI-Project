import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
import torchvision.models as models


df = pd.read_csv("Meat Freshness.multiclass/train/_classes.csv")

print(df.head())
print(df.columns)
# -------- DEFINE WHICH COLUMNS TO USE --------
image_folder = "Meat Freshness.multiclass/train/images/"
filename_column = "filename"
image_paths = [os.path.join(image_folder, f) for f in df[filename_column]]

#image_paths = df["image_path"].tolist()

env_columns = ["filename", "Fresh", "Half-Fresh", "Spoiled"]  
env_data = [env_columns].values.astype(np.float32)

labels = df["freshness_label"].values.astype(np.int64)

# -------- SCALE ENVIRONMENTAL DATA --------
scaler = StandardScaler()
env_data = scaler.fit_transform(env_data)

# -------- TRAIN-TEST SPLIT --------
img_train, img_test, env_train, env_test, y_train, y_test = train_test_split(
    image_paths, env_data, labels, test_size=0.2, random_state=42
)


# ---- image preprocessing ----
def load_and_preprocess_image(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)

class MeatDataset(Dataset):
    def __init__(self, image_paths, environmental_data, labels):
        self.image_paths = image_paths
        self.env_data = environmental_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = load_and_preprocess_image(self.image_paths[idx])
        env_features = torch.tensor(self.env_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, env_features, label



train_dataset = MeatDataset(img_train, env_train, y_train)
test_dataset = MeatDataset(img_test, env_test, y_test)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MeatFreshnessModel(nn.Module):
    def __init__(self, env_size, num_classes):
        super().__init__()

        # Pretrained image encoder (ResNet18 example)
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Identity()           # output = 512 dims

        # Environmental MLP
        self.env_mlp = nn.Sequential(
            nn.Linear(env_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Fusion layer
        self.classifier = nn.Linear(512 + 16, num_classes)

    def forward(self, image, env_features):
        img_feat = self.cnn(image)            # [batch, 512]
        env_feat = self.env_mlp(env_features) # [batch, 16]

        fused = torch.cat([img_feat, env_feat], dim=1)
        return self.classifier(fused)

env_size = env_train.shape[1]     # e.g., 4 features
num_classes = df["freshness_label"].nunique()
env_size = env_train.shape[1]     
num_classes = df["freshness_label"].nunique()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MeatFreshnessModel(env_size, num_classes).to(device)
print("Using device:", device)
loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for images, env_data, labels in train_loader:
    images = images.to(device)
    env_data = env_data.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images, env_data)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()




