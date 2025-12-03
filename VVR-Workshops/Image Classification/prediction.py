import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# -------------------------------------------------
# 1. DEVICE → Auto GPU (CUDA) if available
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# 4. SIMPLE CNN MODEL
# -------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

num_classes = 2
model = CNN(num_classes).to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.to(device)
model.eval()  # set to evaluation mode
print("Model loaded and set to evaluation mode.")

# -------------------------------------------------
# 2. IMAGE TRANSFORMS
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load image
img = Image.open("dog001.jpg").convert("RGB")
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension
img = img.to(device)

with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)
    print("Predicted class:", predicted.item())

if predicted.item() == 0:
    print("The image is classified as: Cat")
else:
    print("The image is classified as: Dog")