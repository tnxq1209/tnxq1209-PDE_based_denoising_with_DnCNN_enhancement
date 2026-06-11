from skimage.restoration import denoise_nl_means

def nlm_filter(image):

    return denoise_nl_means(
        image,
        patch_size=5,
        patch_distance=6,
        h=0.1,
        fast_mode=True
    )