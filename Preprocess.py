from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/MyDrive/ShanghaiTech.zip -d /content/


# ============================================================
#  CSRNet Preprocessing Script for ShanghaiTech Dataset
#  Compatible with Google Colab (stores processed data to Drive)
# ============================================================

# ---------- 1. Mount Google Drive ----------
from google.colab import drive
drive.mount('/content/drive')

import os, cv2, h5py, numpy as np, scipy.io as sio
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# ---------- 2. Configuration ----------
DATASET_PATH = "/content/ShanghaiTech"                    # Extracted dataset in Colab root
OUTPUT_PATH = "/content/drive/MyDrive/Processed_Shanghai"  # Processed dataset will be saved in Drive
TARGET_SIZE = (512, 512)                                  # Resize size (Width, Height)
GAUSSIAN_SIGMA = 15                                       # Smoothing factor
SHOW_VISUAL = False                                       # Debug display off (set True for preview)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ---------- 3. Helper Functions ----------
def preprocess_image(img, target_size):
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

def generate_density_map(image, points, sigma=GAUSSIAN_SIGMA):
    density = np.zeros(image.shape[:2], dtype=np.float32)
    for p in points:
        x = min(int(p[0]), image.shape[1] - 1)
        y = min(int(p[1]), image.shape[0] - 1)
        density[y, x] += 1
    density = gaussian_filter(density, sigma=sigma)
    return density

def generate_pointer_map(image, points, radius=3):
    pointer = np.zeros(image.shape[:2], dtype=np.uint8)
    for p in points:
        x = int(p[0])
        y = int(p[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(pointer, (x, y), radius, 255, -1)
    return pointer

def process_set(part, mode):
    img_dir = os.path.join(DATASET_PATH, f"part_{part}/{mode}_data/images")
# Note spelling for ground truth folder in your image: 'ground-truth'
    gt_dir  = os.path.join(DATASET_PATH, f"part_{part}/{mode}_data/ground-truth")

    out_dir = os.path.join(OUTPUT_PATH, f"part_{part}/{mode}")
    os.makedirs(out_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    for img_name in tqdm(img_files, desc=f"Processing part_{part} {mode}"):
        img_path = os.path.join(img_dir, img_name)
        mat_path = os.path.join(gt_dir, "GT_" + img_name.replace(".jpg", ".mat"))

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipped unreadable image: {img_path}")
            continue

        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]  # (x,y) coordinates

        h, w = img.shape[:2]
        img_proc = preprocess_image(img, TARGET_SIZE)

        scale_x = TARGET_SIZE[0] / w
        scale_y = TARGET_SIZE[1] / h
        points_rescaled = np.zeros_like(points)
        points_rescaled[:, 0] = points[:, 0] * scale_x
        points_rescaled[:, 1] = points[:, 1] * scale_y

        pointer_map = generate_pointer_map(img_proc, points_rescaled, radius=2)
        density_map = generate_density_map(img_proc, points_rescaled, sigma=GAUSSIAN_SIGMA)

        save_path = os.path.join(out_dir, img_name.replace(".jpg", ".h5"))
        with h5py.File(save_path, "w") as hf:
            hf["image"] = img_proc
            hf["density"] = density_map
            hf["pointer"] = pointer_map
            hf["count"] = len(points)

# ---------- 4. Main Processing ----------
if __name__ == "__main__":
    for part in ["A", "B"]:
        for mode in ["train", "test"]:
            process_set(part, mode)

    print(f"\n✅ All done! Processed .h5 files saved to:\n{OUTPUT_PATH}")
