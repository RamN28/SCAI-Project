import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_loader import MeatFreshnessDataset, get_transforms
from models.simple_cnn import SimpleMeatCNN

def train_model():
    # Configuration
    data_path = "data/raw/meat_freshness"  # Update this path
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 25
  # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
  # Data transforms
    train_transform, test_transform = get_transforms()
  # Load dataset
    try:
        dataset = MeatFreshnessDataset(data_path, transform=train_transform)
        print(f"Loaded {len(dataset)} images")
        print(f"Classes: {dataset.classes}")

        # Split dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Apply test transform to test dataset
        test_dataset.dataset.transform = test_transform
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your data path and structure")
        return

        # Model, loss, optimizer
        model = SimpleMeatCNN(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

         # Training history
        train_losses = []
        train_accs = []
        test_accs = []

         # Training loop
        for epoch in range(num_epochs):
          model.train()
          running_loss = 0.0
          correct = 0
          total = 0
          progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

             # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct/total:.2f}%'
            })

        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Test the model
        model.eval()
        test_correct = 0
        test_total = 0

         with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
          test_acc = 100 * test_correct / test_total
          test_accs.append(test_acc)

          print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%')
     # Save model
    torch.save(model.state_dict(), 'models/meat_freshness_model.pth')
    print("Model saved!")

    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
if __name__ == "__main__":
    train_model()
