import torch
import cv2
import numpy as np
from dncnn_model import DnCNN
from metrics import calculate_psnr, calculate_ssim, calculate_mse

# Load model
device = torch.device("cpu")
model = DnCNN(channels=1).to(device)
model.load_state_dict(torch.load("dncnn/checkpoints/dncnn_final.pth", map_location=device))
model.eval()

# Load clean image
img = cv2.imread("images/a8.jpg", cv2.IMREAD_GRAYSCALE)
clean = img.astype(np.float32) / 255.0

# Add noise (same as training)
noise = np.random.randn(*clean.shape) * (25 / 255.0)
noisy = clean + noise
noisy = np.clip(noisy, 0, 1)

# DnCNN inference
noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    denoised = model(noisy_tensor)

denoised = denoised.squeeze().cpu().numpy()
denoised = np.clip(denoised, 0, 1)

# Metrics
mse = calculate_mse(clean, denoised)
psnr = calculate_psnr(clean, denoised)
ssim = calculate_ssim(clean, denoised)

print(f"MSE  : {mse:.6f}")
print(f"PSNR : {psnr:.2f} dB")
print(f"SSIM : {ssim:.4f}")
