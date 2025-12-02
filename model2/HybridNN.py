import torch.nn as nn

class FreshnessClassifier(nn.Module):
  def __init__(self):
        super().__init__()
        # CNN for images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.env_net = nn.Sequential(
            nn.Linear(3, 16),  # temp, time, location
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * ? + 8, 32),  # You'll need to calculate the CNN output size
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 classes: fresh, half-fresh, spoiled
        )

  def forward(self, image, env_data):
        img_features = self.conv_layers(image)
        env_features = self.env_net(env_data)
        combined = torch.cat([img_features, env_features], dim=1)
        return self.classifier(combined)
