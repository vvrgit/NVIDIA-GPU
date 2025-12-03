import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 2. LOAD DATA
# -----------------------------
df = pd.read_excel("India NREL DATASET.xlsx")  # <-- your file

# Check missing values
print("Missing values before:", df.isnull().sum())

# ---------------------------------------
# 2. HANDLE NULL VALUES
# ---------------------------------------

# OPTION 1 → Drop rows with ANY null value
# df = df.dropna()

# OPTION 2 → Fill nulls with column mean (recommended)
df = df.fillna(df.mean(numeric_only=True))

# OPTION 3 → Interpolate missing values (time-series friendly)
# df = df.interpolate(method='linear')

print("Missing values after:", df.isnull().sum())



X = df[['height',
        'air pressure  (Pa)',
        'wind direction  (deg)',
        'temperature  (C)']].values

y = df[['wind speed']].values       # <-- target column name

# -----------------------------
# 3. NORMALIZE FEATURES
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# -----------------------------
# 5. DEFINE ANN MODEL
# -----------------------------
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

model = WindANN().to(device)
print(model)

# -----------------------------
# 6. LOSS + OPTIMIZER
# -----------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 7. TRAIN LOOP
# -----------------------------
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.4f}")

# -----------------------------
# 8. SAVE MODEL (SAFE)
# -----------------------------
torch.save(model.state_dict(), "wind_ann_weights.pth")
print("Model saved successfully!")
