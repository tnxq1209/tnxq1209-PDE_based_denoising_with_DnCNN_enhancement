import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    return structural_similarity(img1, img2, data_range=1.0)
