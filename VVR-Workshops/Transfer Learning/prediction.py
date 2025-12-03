import torch
from torchvision import models, transforms
from PIL import Image

# ----------------------------
# 1️⃣ Set device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# 2️⃣ Load the pretrained model architecture
# ----------------------------
model = models.resnet18(pretrained=False)  # don't load ImageNet weights
num_classes = 2  # change this to your dataset's number of classes
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load("resnet18_transfer.pth", map_location=device))
model = model.to(device)
model.eval()  # set model to evaluation mode

# ----------------------------
# 3️⃣ Prepare image transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # same as during training
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# 4️⃣ Load and preprocess the image
# ----------------------------
image_path = "dog001.jpg"  # replace with your image path
img = Image.open(image_path).convert("RGB")
img = transform(img)
img = img.unsqueeze(0)  # add batch dimension
img = img.to(device)

# ----------------------------
# 5️⃣ Predict class
# ----------------------------
with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

# Print predicted class index
print("Predicted class index:", predicted.item())

# Optional: if you have class names
class_names = ["Cat", "Dog"]  # adjust accordingly
print("Predicted class name:", class_names[predicted.item()])
