import h5py
from PIL import Image
import numpy as np
import cv2

def load_data(h5_path, train=True):
    with h5py.File(h5_path, 'r') as hf:
        img_np = np.array(hf['image'])
        density = np.array(hf['density'])

    img_np = np.squeeze(img_np)
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError(f"Image array has invalid shape: {img_np.shape}")

    img = Image.fromarray(img_np.astype('uint8'))

    # Optional data augmentation here

    density = cv2.resize(density, (density.shape[1]//8, density.shape[0]//8), interpolation=cv2.INTER_CUBIC) * 64

    return img, density
