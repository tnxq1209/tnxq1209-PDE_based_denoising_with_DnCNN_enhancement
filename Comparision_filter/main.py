import pandas as pd
import numpy as np

from metrics import calculate_metrics

from Gaussian import gaussian_filter
from nlm import nlm_filter
from bm3d import bm3d_filter
from Anisotropic_function import anisotropic_diffusion
from proposed_filter import filter_name

from noise import add_poisson_gaussian_noise
from image_loader import load_image


image = load_image("test_image.png")

noise_levels = [5, 10, 15, 20, 25, 30]

results = []

for noise_percent in noise_levels:

    noisy = add_poisson_gaussian_noise(
        image,
        noise_percent
    )

    filters = {
        "Gaussian": gaussian_filter(noisy),
        "NLM": nlm_filter(noisy),
        "BM3D": bm3d_filter(noisy),
        "Proposed": anisotropic_diffusion(noisy)
    }

    for name, output in filters.items():

        psnr, ssim = calculate_metrics(
            image,
            output
        )

        results.append({
            "Noise %": noise_percent,
            "Filter": name,
            "PSNR": round(psnr, 3),
            "SSIM": round(ssim, 4)
        })

df = pd.DataFrame(results)

df.to_excel(
    "filter_comparison.xlsx",
    index=False
)

print(df)
print("\nSaved to filter_comparison.xlsx")