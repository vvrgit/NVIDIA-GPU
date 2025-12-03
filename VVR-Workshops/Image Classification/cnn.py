import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------------------------------
# 1. DEVICE → Auto GPU (CUDA) if available
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# 2. IMAGE TRANSFORMS
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -------------------------------------------------
# 3. LOAD TRAIN + VALIDATION DATA
# Folder structure:
# data/
#   train/class1/*.jpg
#   train/class2/*.jpg
#   val/class1/*.jpg
#   val/class2/*.jpg
# -------------------------------------------------
train_data = datasets.ImageFolder("dataset/train", transform=transform)
val_data   = datasets.ImageFolder("dataset/test",   transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False)

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

num_classes = len(train_data.classes)
model = CNN(num_classes).to(device)

# -------------------------------------------------
# 5. LOSS + OPTIMIZER
# -------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------------------------
# 6. TRAIN + VALIDATION LOOP
# -------------------------------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Train Loss={train_loss:.4f}, "
          f"Val Loss={val_loss:.4f}, "
          f"Val Acc={val_accuracy:.2f}%")

print("\nTraining Complete!")

# -------------------------------------------------
# 7. SAVE MODEL
# -------------------------------------------------
torch.save(model.state_dict(), "cnn_model.pth")
print("Model saved as cnn_model.pth")
