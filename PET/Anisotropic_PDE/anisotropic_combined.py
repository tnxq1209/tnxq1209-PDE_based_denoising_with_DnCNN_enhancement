import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(
    "/home/winters/Projects/Final_year_project/PET/Anisotropic_PDE/1-0077.png",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError("Image not found")
image = image.astype(np.float32)
image_norm = image / image.max()
peak = 50
noisy = np.random.poisson(
    image_norm * peak
).astype(np.float32) / peak

noisy *= image.max()

noisy = np.clip(
    noisy,
    image.min(),
    image.max()
)

def add_gaussian_noise(noisy, noise_percent):

    signal_rms = np.sqrt(np.mean(noisy**2))

    sigma = (noise_percent / 100.0) * signal_rms

    noise = np.random.normal(
        0,
        sigma,
        noisy.shape
    ).astype(np.float32)

    noisy = noisy + noise

    noisy = np.clip(
        noisy,
        image.min(),
        image.max()
    )

    return noisy
noisy = add_gaussian_noise(noisy, noise_percent=15)

#This guarantees that the noisy image remains in the same intensity scale as the original.

def anisotropic_diffusion(img,
                          iterations=10,
                          kappa=30,
                          gamma=0.08):

    img = img.copy().astype(np.float32)

    for _ in range(iterations):

        north = np.roll(img, -1, axis=0) - img
        south = np.roll(img, 1, axis=0) - img
        east  = np.roll(img, -1, axis=1) - img
        west  = np.roll(img, 1, axis=1) - img

        c_n = np.exp(-(north / kappa) ** 2)
        c_s = np.exp(-(south / kappa) ** 2)
        c_e = np.exp(-(east  / kappa) ** 2)
        c_w = np.exp(-(west  / kappa) ** 2)

        img += gamma * (
            c_n * north +
            c_s * south +
            c_e * east +
            c_w * west
        )

    return img
filtered = anisotropic_diffusion(
    noisy,
    iterations=15,
    kappa=65,
    gamma=0.075
)
median = cv2.medianBlur(filtered.astype(np.uint8), 3).astype(np.float32)
filtered = 0.7 * filtered + 0.3 * median
filtered = np.clip(
    filtered,
    image.min(),
    image.max()
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(filtered, cmap='gray')
plt.title("Filtered")
plt.axis("off")

plt.tight_layout()
plt.savefig("comparison_combined.png", dpi=300)
print("IMAGE FILTERED AND SAVED AS \"comparison_combined.png\".")
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error
)

# ----------------------------
# Ensure same size

reference = image
if reference.shape != filtered.shape:
    filtered = cv2.resize(
        filtered,
        (reference.shape[1], reference.shape[0])
    )

# ----------------------------
# Metrics
# ----------------------------

mse = mean_squared_error(reference, filtered)

psnr_n = peak_signal_noise_ratio(
    reference,
    noisy,
    data_range = reference.max() - reference.min()
)

ssim_n = structural_similarity(
    reference,
    noisy,
    data_range = reference.max() - reference.min()
)
psnr_f = peak_signal_noise_ratio(
    reference,
    filtered,
    data_range = reference.max() - reference.min()
)

ssim_f = structural_similarity(
    reference,
    filtered,
    data_range = reference.max() - reference.min()
)
#noisy
print(f"PSNR Noisy: {psnr_n:.4f} dB")
print(f"SSIM Noisy: {ssim_n:.4f}")
#filtered
print(f"PSNR Filtered: {psnr_f:.4f} dB")
print(f"SSIM Filtered: {ssim_f:.4f}")

#Relative RMS Noise (%)

noise_power = np.mean((noisy-image)**2)
signal_power = np.mean(image**2)

noise_percentage = (
    np.sqrt(noise_power)
    / np.sqrt(signal_power)
) * 100

print(f"Noise Percentage: {noise_percentage:.2f}%")