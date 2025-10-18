import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# -----------------------------
# CONFIGURATION: point to file in Drive
# -----------------------------
H5_PATH = "/content/drive/MyDrive/Processed_Shanghai/part_B/test/IMG_1.h5"

# -----------------------------
# VISUALIZATION FUNCTION
# -----------------------------
def visualize_processed_sample(h5_path):
    if not os.path.exists(h5_path):
        print(f"❌ File not found: {h5_path}")
        return

    # ---- Load processed data ----
    with h5py.File(h5_path, "r") as f:
        image = f["image"][:]
        density = f["density"][:]
        pointer = f["pointer"][:]
        count = f["count"][()]
    
    print(f"✅ Loaded: {os.path.basename(h5_path)}")
    print(f"👥 Ground Truth Count: {int(count)}")
    print(f"📏 Image Shape: {image.shape}")

    # ---- Prepare overlays ----
    img_rgb = (image * 255).astype(np.uint8)
    pointer_overlay = img_rgb.copy()
    pointer_overlay[pointer > 0] = [0, 0, 255]  # Red dots for individuals

    # ---- Visualize ----
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(pointer_overlay, cv2.COLOR_BGR2RGB))
    plt.title("Pointer Map (Individual Head Dots)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(density, cmap='jet')
    plt.title(f"Density Map (Sum ≈ {density.sum():.1f})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# -----------------------------
# MAIN EXECUTION
# -----------------------------
visualize_processed_sample(H5_PATH)
