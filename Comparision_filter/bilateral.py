import cv2

def bilateral_filter(image,
                     diameter=9,
                     sigma_color=75,
                     sigma_space=75):

    img_uint8 = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    filtered = cv2.bilateralFilter(
        img_uint8,
        diameter,
        sigma_color,
        sigma_space
    )

    return filtered.astype(float)