import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import sys
sys.path.append("/home/winters/Projects/Final_year_project")
from PET.Anisotropic_PDE.anisotropic_combined import anisotropic_diffusion

class DnCNNCleanImageDataset(Dataset):
    def __init__(
        self,
        clean_dir,
        patch_size=64,
        noise_type="gaussian",
        sigma=25,
        patches_per_image=100
    ):
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.noise_type = noise_type
        self.sigma = sigma
        self.patches_per_image = patches_per_image

        self.files = sorted([
            f for f in os.listdir(clean_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        # Artificially enlarge dataset size
        return len(self.files) * self.patches_per_image

    def __getitem__(self, idx):
        # Map index to image
        img_idx = idx % len(self.files)
        path = os.path.join(self.clean_dir, self.files[img_idx])
        if idx < 5:
            print(f"IDX={idx}, IMAGE={self.files[img_idx]}")

        # Load clean image
        clean = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        clean = clean.astype(np.float32) / 255.0

        H, W = clean.shape
        ps = self.patch_size

        # Random crop
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        clean_patch = clean[top:top + ps, left:left + ps]

        # Convert to tensor [1, H, W]
        clean_patch = torch.from_numpy(clean_patch).unsqueeze(0)

        # Add noise
        # Generate PET noise
        noisy_patch = add_pet_noise(clean_patch)

        # Run anisotropic diffusion
        filtered_patch = anisotropic_diffusion(noisy_patch)

        return filtered_patch, clean_patch
