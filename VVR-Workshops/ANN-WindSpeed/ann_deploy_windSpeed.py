import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- same model architecture ----
class WindANN(nn.Module):
    def __init__(self):
        super(WindANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# Load model weights
model = WindANN().to(device)
model.load_state_dict(torch.load("wind_ann_weights.pth", map_location=device))
model.eval()

# Example input sample (normalized the same way!)
sample = [[
    73.0,    # height
    101325,  # air pressure (Pa)
    220,     # wind direction (deg)
    25.3     # temperature (C)
]]

x = torch.tensor(sample, dtype=torch.float32).to(device)

with torch.no_grad():
    pred = model(x)

print("Predicted Wind Speed:", float(pred.item()))
