import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#sample dataset
import numpy as np

# Example: 10000 samples, 20 features
X = np.random.rand(10000, 20)

X = torch.tensor(X, dtype=torch.float32).to(device)

#autoencoder model
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=20, encoding_dim=5):
        super(AutoEncoder, self).__init__()
        
        # Encoder (Compression)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder (Reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    
#Model, Loss, Optimizer (GPU)
model = AutoEncoder(input_dim=20, encoding_dim=5).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Train Model on GPU
epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    
    encoded, decoded = model(X)
    
    loss = criterion(decoded, X)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")


#Data Compression and Reconstruction
with torch.no_grad():
    compressed_data = model.encoder(X)

print("Original shape:", X.shape)
print("Compressed shape:", compressed_data.shape)


with torch.no_grad():
    reconstructed_data = model.decoder(compressed_data)
print("Reconstructed shape:", reconstructed_data.shape)

torch.save(model.state_dict(), "autoencoder_gpu.pth")
