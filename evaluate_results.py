from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =========================================================
# ORIGINAL CLASS FOLDER
# =========================================================
class_name = "squamouscellcarcinoma"
original_root = Path(f"Noise_Model/Images/PT/PT/{class_name}")

# Root folders that contain 5%, 10%, 15%, ...
noisy_root = Path("Noise_Model/INPUT_PT")
filtered_root = Path("Noise_Model/OUTPUT_PT")

# Output folder
base_output_dir = Path("combined result comparison")
base_output_dir.mkdir(parents=True, exist_ok=True)

# Noise levels to process
noise_levels = ["5%", "10%", "15%", "20%", "25%", "30%"]

# =========================================================
# HELPERS
# =========================================================
def is_image_file(p: Path):
    return p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def collect_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.is_file() and is_image_file(p)])

def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def compute_metrics(original, other):
    if original.shape != other.shape:
        other = cv2.resize(other, (original.shape[1], original.shape[0]))
    psnr = peak_signal_noise_ratio(original, other, data_range=255)
    ssim = structural_similarity(original, other, data_range=255)
    return psnr, ssim, other

def find_match(original_path: Path, search_root: Path):
    """
    Try to match by same relative path first.
    If not found, search by filename anywhere under search_root.
    """
    try:
        rel = original_path.relative_to(original_root)
        candidate = search_root / rel
        if candidate.exists():
            return candidate
    except Exception:
        pass

    matches = [p for p in search_root.rglob(original_path.name) if p.is_file() and is_image_file(p)]
    if len(matches) >= 1:
        return matches[0]
    return None

# =========================================================
# ORIGINAL IMAGES
# =========================================================
original_images = collect_images(original_root)
if not original_images:
    raise ValueError(f"No images found in: {original_root}")

print(f"Found {len(original_images)} original images in {original_root}")

# =========================================================
# PROCESS EACH NOISE LEVEL
# =========================================================
for level in noise_levels:
    noisy_level_root = noisy_root / level
    filtered_level_root = filtered_root / level

    if not noisy_level_root.exists():
        print(f"Skipping {level}: missing noisy folder {noisy_level_root}")
        continue
    if not filtered_level_root.exists():
        print(f"Skipping {level}: missing filtered folder {filtered_level_root}")
        continue

    level_output_dir = base_output_dir / level
    level_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    count = 0

    print(f"\nProcessing noise level: {level}")

    for orig_path in original_images[:50]:  # first 50 images
        noisy_path = find_match(orig_path, noisy_level_root)
        filt_path = find_match(orig_path, filtered_level_root)

        if noisy_path is None or filt_path is None:
            print(f"Skipping {orig_path.name} for {level}: noisy/filtered file not found.")
            continue

        orig = load_gray(orig_path)
        noisy = load_gray(noisy_path)
        filt = load_gray(filt_path)

        psnr_noisy, ssim_noisy, noisy_resized = compute_metrics(orig, noisy)
        psnr_filt, ssim_filt, filt_resized = compute_metrics(orig, filt)

        results.append({
            "noise_level": level,
            "filename": orig_path.name,
            "relative_path": str(orig_path.relative_to(original_root)),
            "noisy_path": str(noisy_path),
            "filtered_path": str(filt_path),
            "psnr_noisy": psnr_noisy,
            "ssim_noisy": ssim_noisy,
            "psnr_filtered": psnr_filt,
            "ssim_filtered": ssim_filt
        })

        # Create one comparison figure per image
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        axes[0].imshow(orig, cmap="gray")
        axes[0].set_title("Original", fontsize=12)

        axes[1].imshow(noisy_resized, cmap="gray")
        axes[1].set_title(f"Noisy ({level})", fontsize=12)
        axes[1].text(
            0.02, 0.98,
            f"PSNR: {psnr_noisy:.2f}\nSSIM: {ssim_noisy:.4f}",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3)
        )

        axes[2].imshow(filt_resized, cmap="gray")
        axes[2].set_title(f"Filtered ({level})", fontsize=12)
        axes[2].text(
            0.02, 0.98,
            f"PSNR: {psnr_filt:.2f}\nSSIM: {ssim_filt:.4f}",
            transform=axes[2].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3)
        )

        for ax in axes:
            ax.axis("off")

        rel_name = orig_path.relative_to(original_root).as_posix().replace("/", "_")
        save_path = level_output_dir / f"comparison_{count+1}_{rel_name}.png"

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {save_path}")
        count += 1

    # Save CSV for this noise level
    csv_path = level_output_dir / f"pt_metrics_{class_name}_{level}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV: {csv_path}")
    print(f"Done for {level}. Saved {count} images.\n")

print("All done.")