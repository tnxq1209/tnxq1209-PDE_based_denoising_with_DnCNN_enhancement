# image_loader.py

import os
import cv2
import numpy as np


VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


def load_images_from_folder(folder_path):

    images = []

    for filename in sorted(os.listdir(folder_path)):

        if filename.lower().endswith(VALID_EXTENSIONS):

            image_path = os.path.join(
                folder_path,
                filename
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:

                images.append({
                    "name": filename,
                    "image": image.astype(np.float32)
                })

    return images