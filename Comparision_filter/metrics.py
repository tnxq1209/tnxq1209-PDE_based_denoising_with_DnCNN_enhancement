from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

def calculate_metrics(original, denoised):

    psnr = peak_signal_noise_ratio(
        original,
        denoised,
        data_range=original.max() - original.min()
    )

    ssim = structural_similarity(
        original,
        denoised,
        data_range=original.max() - original.min()
    )

    return psnr, ssim