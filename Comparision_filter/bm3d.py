import bm3d

def bm3d_filter(image,
                sigma=25):

    return bm3d.bm3d(
        image,
        sigma_psd=sigma/255.0
    )