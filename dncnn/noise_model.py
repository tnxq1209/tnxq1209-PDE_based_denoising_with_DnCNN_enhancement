import torch

def add_gaussian_noise(img, sigma=25):
    noise = torch.randn_like(img) * (sigma / 255.0)
    return img + noise
