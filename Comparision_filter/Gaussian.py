import cv2

def gaussian_filter(image, sigma=1.5):
    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )
