import torch
from torch.utils.data import DataLoader
from dncnn_model import DnCNN
from dncnn_dataset import DnCNNCleanImageDataset
from tqdm import tqdm
import os

# Config
CLEAN_DIR = "images"
EPOCHS = 30
BATCH_SIZE = 1
LR = 1e-3

# Dataset & Loader
dataset = DnCNNCleanImageDataset(
    clean_dir="images",
    patch_size=64,
    noise_type="gaussian",
    sigma=25,
    patches_per_image=100
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
device = torch.device("cpu")
model = DnCNN(channels=1).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Checkpoint dir
os.makedirs("checkpoints", exist_ok=True)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

    for filtered, clean in progress:
        filtered = filtered.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        denoised = model(filtered)
        loss = criterion(denoised, clean) # Target is the clean image

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.6f}")

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")

# Optional final save
torch.save(
    model.state_dict(),
    "dncnn/checkpoints/dncnn_final.pth"
)

print("Training completed and model saved")
