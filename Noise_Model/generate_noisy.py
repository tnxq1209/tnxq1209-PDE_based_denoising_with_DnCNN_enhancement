"""
generate_noisy.py
-----------------
Adds combined Poisson + Gaussian noise to PET/CT images at multiple
noise levels and organises them into output folders.

Input folder layout
───────────────────
Images/
    PT/   ← PET scan images  (*.png)
    CT/   ← CT  scan images  (*.png)

Output folder layout
────────────────────
INPUT_PT/
    5%/    10%/    15%/    20%/    25%/    30%/
INPUT_CT/
    5%/    10%//   15%/    20%/    25%/    30%/

Noise model (combined, matching anisotropic_combined.py)
─────────────────────────────────────────────────────────
  1. Poisson shot noise  — simulates photon-counting noise in PET/CT.
     The noise level % controls the Poisson peak parameter:
         peak = 10000 / noise_percent²
     Higher % → lower peak → more Poisson noise.

  2. Gaussian noise      — additive electronic / system noise.
     sigma = (noise_percent / 100) × RMS of Poisson-noisy image.

  The clipping after each step keeps intensity in the original image range.

Usage
─────
    python generate_noisy.py                        # uses defaults
    python generate_noisy.py --input  Images        \
                              --out-pt INPUT_PT      \
                              --out-ct INPUT_CT      \
                              --levels 5 10 15 20 25 30
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

os.chdir(Path(__file__).parent)


# ── Noise functions ───────────────────────────────────────────────────────────

def add_poisson_noise(image: np.ndarray, noise_percent: float) -> np.ndarray:
    """
    Apply Poisson (shot) noise.

    peak controls how much Poisson noise is injected:
        higher peak  → less noise  (more photons)
        lower  peak  → more noise  (fewer photons)

    We map noise_percent → peak so that 5% is nearly clean and
    30% is heavily degraded, matching typical PET low-count scenarios.

    Args:
        image        : float32 input, any intensity range.
        noise_percent: target noise level in percent (5 – 30).

    Returns:
        float32 Poisson-noisy image clipped to original intensity range.
    """
    img_min = float(image.min())
    img_max = float(image.max())

    if img_max == 0:
        return image.copy()

    # Normalise to [0, 1]
    norm = image / img_max

    # peak: number of "photon counts" per unit intensity
    # Formula gives peak≈40000 at 5% and peak≈1111 at 30%
    peak = 10_000.0 / (noise_percent ** 2)

    noisy_norm = np.random.poisson(norm * peak).astype(np.float32) / peak

    # Restore original scale and clip
    noisy = noisy_norm * img_max
    return np.clip(noisy, img_min, img_max)


def add_gaussian_noise(image: np.ndarray, noise_percent: float) -> np.ndarray:
    """
    Add zero-mean Gaussian noise whose std = (noise_percent/100) × RMS(image).

    Args:
        image        : float32 input (already Poisson-noisy).
        noise_percent: target noise level in percent.

    Returns:
        float32 image with Gaussian noise added, clipped to original range.
    """
    img_min = float(image.min())
    img_max = float(image.max())

    signal_rms = np.sqrt(np.mean(image ** 2))
    sigma = (noise_percent / 100.0) * signal_rms

    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image + noise
    return np.clip(noisy, img_min, img_max)


def add_combined_noise(image: np.ndarray, noise_percent: float) -> np.ndarray:
    """
    Full combined noise: Poisson first, then Gaussian on top.

    Args:
        image        : float32 clean image.
        noise_percent: target noise level in percent (e.g. 15 for 15%).

    Returns:
        float32 noisy image.
    """
    noisy = add_poisson_noise(image, noise_percent)
    noisy = add_gaussian_noise(noisy, noise_percent)
    return noisy


# ── Core processing ───────────────────────────────────────────────────────────

def process_modality(
    src_dir: Path,
    dst_root: Path,
    noise_levels: list[int],
    modality: str,
) -> None:
    """
    Process all PNG images in src_dir for every noise level.

    Args:
        src_dir     : e.g. Images/PT
        dst_root    : e.g. INPUT_PT
        noise_levels: list of integer percentages, e.g. [5, 10, 15, 20, 25, 30]
        modality    : 'PT' or 'CT' (used only for console output)
    """
    img_paths = sorted([
        p for p in src_dir.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"}
           and p.is_file()
    ])

    if not img_paths:
        print(f"  [WARN] No images found in {src_dir}")
        return

    print(f"\n[{modality}] Found {len(img_paths)} image(s) in {src_dir}")

    # Pre-create all output subdirs
    level_dirs: dict[int, Path] = {}
    for lvl in noise_levels:
        d = dst_root / f"{lvl}%"
        d.mkdir(parents=True, exist_ok=True)
        level_dirs[lvl] = d

    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] Cannot read {img_path.name}, skipping.")
            continue

        img_f = img.astype(np.float32)

        for lvl in noise_levels:
            noisy_f = add_combined_noise(img_f, noise_percent=float(lvl))
            noisy_u8 = np.clip(noisy_f, 0, 255).astype(np.uint8)

            out_path = level_dirs[lvl] / img_path.name
            cv2.imwrite(str(out_path), noisy_u8)

        print(f"  {img_path.name}  → {', '.join(str(l)+'%' for l in noise_levels)}")

    print(f"[{modality}] Done → {dst_root}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate combined Poisson+Gaussian noisy PET/CT images at multiple levels"
    )
    p.add_argument(
        "--input", default="Images",
        help="Root folder containing PT/ and CT/ subfolders (default: Images)"
    )
    p.add_argument(
        "--out-pt", default="INPUT_PT",
        help="Output root folder for PET noisy images (default: INPUT_PT)"
    )
    p.add_argument(
        "--out-ct", default="INPUT_CT",
        help="Output root folder for CT  noisy images (default: INPUT_CT)"
    )
    p.add_argument(
        "--levels", nargs="+", type=int,
        default=[5, 10, 15, 20, 25, 30],
        help="Noise levels in percent (default: 5 10 15 20 25 30)"
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (optional)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[Info] Random seed set to {args.seed}")

    input_root = Path(args.input)
    out_pt     = Path(args.out_pt)
    out_ct     = Path(args.out_ct)
    levels     = sorted(set(args.levels))

    print(f"[Info] Input root : {input_root}")
    print(f"[Info] Noise levels: {levels}")
    print(f"[Info] Noise model : Combined Poisson + Gaussian\n")

    pt_dir = input_root / "PT"
    ct_dir = input_root / "CT"

    if not pt_dir.exists():
        print(f"[WARN] PT folder not found: {pt_dir}")
    else:
        process_modality(pt_dir, out_pt, levels, modality="PT")

    if not ct_dir.exists():
        print(f"[WARN] CT folder not found: {ct_dir}")
    else:
        process_modality(ct_dir, out_ct, levels, modality="CT")

    print("\n[Done] All noisy images generated.")
    print(f"  INPUT_PT/  →  {out_pt.resolve()}")
    print(f"  INPUT_CT/  →  {out_ct.resolve()}")


if __name__ == "__main__":
    main()