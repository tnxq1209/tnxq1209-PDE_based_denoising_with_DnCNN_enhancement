import cv2
import numpy as np

def anisotropic_diffusion(img,
                        iterations=10,
                        kappa=30,
                        gamma=0.08):

    img = img.copy().astype(np.float32)

    for _ in range(iterations):

        north = np.roll(img, -1, axis=0) - img
        south = np.roll(img, 1, axis=0) - img
        east  = np.roll(img, -1, axis=1) - img
        west  = np.roll(img, 1, axis=1) - img

        c_n = np.exp(-(north / kappa) ** 2)
        c_s = np.exp(-(south / kappa) ** 2)
        c_e = np.exp(-(east  / kappa) ** 2)
        c_w = np.exp(-(west  / kappa) ** 2)

        img += gamma * (
            c_n * north +
            c_s * south +
            c_e * east +
            c_w * west
        )

    return img