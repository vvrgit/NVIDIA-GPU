import torch
import torch.nn as nn

# -----------------------------
# 1. DEVICE (GPU if available)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. DEFINE SAME MODEL STRUCTURE
# -----------------------------
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)   # regression
        )
       
    def forward(self, x):
        return self.nn(x)

# -----------------------------
# 3. LOAD SAVED MODEL
# -----------------------------
model = ANNModel().to(device)
model.load_state_dict(torch.load("ann_model.pth", map_location=device))
model.eval()

print("Model loaded successfully!")

# -----------------------------
# 4. MAKE PREDICTION
# -----------------------------
# Example new data → 10 numerical features
new_sample = [0.5, 1.1, -0.3, 0.7, 2.0, -0.9, 1.2, 0.4, -0.5, 0.8]

# Convert to tensor
input_tensor = torch.tensor([new_sample], dtype=torch.float32).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)

print("Prediction:", output.item())