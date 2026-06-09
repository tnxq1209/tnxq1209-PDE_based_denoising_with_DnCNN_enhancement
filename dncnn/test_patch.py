import torch
import cv2
import numpy as np
import random
from dncnn_model import DnCNN
from skimage.metrics import structural_similarity as ssim

# --------------------
# Load model
# --------------------
device = torch.device("cpu")
model = DnCNN(channels=1).to(device)
model.load_state_dict(torch.load("dncnn/checkpoints/dncnn_final.pth", map_location=device))
model.eval()

# --------------------
# Load clean image
# --------------------
img = cv2.imread("images/h99.jpg", cv2.IMREAD_GRAYSCALE)
clean = img.astype(np.float32) / 255.0

H, W = clean.shape
ps = 64

# --------------------
# Take RANDOM patch
# --------------------
top = random.randint(0, H - ps)
left = random.randint(0, W - ps)
clean_patch = clean[top:top+ps, left:left+ps]

# --------------------
# Add noise
# --------------------
noise = np.random.randn(ps, ps) * (25 / 255.0)
noisy_patch = np.clip(clean_patch + noise, 0, 1)

# --------------------
# DnCNN inference
# --------------------
noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    denoised_patch = model(noisy_tensor)

denoised_patch = denoised_patch.squeeze().cpu().numpy()
denoised_patch = np.clip(denoised_patch, 0, 1)

# --------------------
# SSIM (PATCH LEVEL)
# --------------------
ssim_val = ssim(clean_patch, denoised_patch, data_range=1.0)

print(f"Patch SSIM: {ssim_val:.4f}")
