# mainstuff.py - Simple test to check if everything works
import torch
from CNNModel import SimpleMeatCNN

def test_model():
    # Test if model can be created
    model = SimpleMeatCNN(num_classes=3)
    print("✓ Model created successfully!")
    
    # Test a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # batch_size=1, channels=3, height=224, width=224
    output = model(dummy_input)
    print(f"✓ Model forward pass works! Output shape: {output.shape}")
    print(f"✓ Output: {output}")
    
    return True

if __name__ == "__main__":
    test_model()
