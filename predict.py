import torch
from simpleNN import myNN
from PIL import Image
from torchvision import transforms

classes = ["Fresh", "Half Fresh", "Spoiled"]

model = myNN()
model.load_state_dict(torch.load("meat_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(path):
    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    print("Prediction:", classes[predicted.item()])


img_path = input("Enter image path: ")
predict_image(img_path)
