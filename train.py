import torch
import torch.nn as nn
import torch.optim as optim
from simpleNN import myNN
from datasets import train_loader, class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = myNN().to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

# Save model
torch.save(model.state_dict(), "meat_model.pth")
print("Model saved as meat_model.pth")
