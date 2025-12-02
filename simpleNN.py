import torch
import torch.nn as nn #neural network section of torch

class myNN(nn.Module):

    def __init__(self): #for layers
        super(myNN, self).__init__()

        # use convolutional layers (for images)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)   # 3 color channels
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Reduce image size
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 56 * 56, 64)  # after pooling twice
        self.fc2 = nn.Linear(64, 3)             # 3 classes output
        
        self.relu = nn.ReLU()  #activation! Defines RELU function, more common 
        
    # can also use nn.sigmoind 
        
    #now need to call all the methods

    def forward(self, x): #for activation

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten tensor
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

