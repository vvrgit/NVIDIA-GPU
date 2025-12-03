import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# 1. DEVICE CONFIGURATION
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. SAMPLE NUMERICAL DATA
# -----------------------------
# X shape: samples × features
# y shape: samples × 1
X = torch.randn(1000, 10)  # 1000 samples, 10 numerical features
y = torch.randn(1000, 1)   # Regression output

# Move data to GPU
X = X.to(device)
y = y.to(device)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# -----------------------------
# 3. DEFINE ANN MODEL
# -----------------------------
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # regression output
        )
       
    def forward(self, x):
        return self.nn(x)

model = ANNModel().to(device)

# -----------------------------
# 4. LOSS AND OPTIMIZER
# -----------------------------
criterion = nn.MSELoss()            # For regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
epochs = 3
for epoch in range(epochs):
    epoch_loss = 0
   
    for batch_X, batch_y in loader:
        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
       
        loss.backward()
        optimizer.step()
       
        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

# -----------------------------
# 6. SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "ann_model.pth")
print("Model saved successfully!")

torch.save(model, "ann_model_architecture.pth")
print("Model saved successfully!")