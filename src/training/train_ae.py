import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_autoencoder import LSTMAutoencoder

# Load dataset
data = np.load("data/processed/train_sequences.npy")

print("Dataset shape:", data.shape)

# Convert to tensor
data = torch.tensor(data)

dataset = TensorDataset(data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = LSTMAutoencoder()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
EPOCHS = 30

for epoch in range(EPOCHS):
    total_loss = 0

    for (x,) in loader:
        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, x)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "outputs/lstm_autoencoder.pth")

print("Model saved!")