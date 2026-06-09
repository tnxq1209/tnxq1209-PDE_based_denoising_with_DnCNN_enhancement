import os
import cv2
import torch
import numpy as np

def load_images_from_folder(folder):
    images = []
    names = []

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            images.append(img)
            names.append(filename)

    return images, names

