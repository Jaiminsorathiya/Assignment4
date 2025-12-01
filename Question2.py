import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

# ==============================
# 1) Here I specify the folders that contain my BSDS300 images
# ==============================
TRAIN_DIR = r"C:\Users\91635\Downloads\snerf_project\BSDS300\images\train"
TEST_DIR  = r"C:\Users\91635\Downloads\snerf_project\BSDS300\images\test"

# Here I decide whether to use only the train set or both train and test
IMAGE_DIRS = [TRAIN_DIR, TEST_DIR]   # I can remove TEST_DIR if I want only train

MAX_IMAGES = 3   # this is the total number of images I process (across all folders)


# ==============================
# 2) Here I build 5D feature vectors for each pixel: [row, col, R, G, B]
# ==============================
def build_features(img):
    img = img.astype(np.float64)
    H, W, C = img.shape  # I expect images to be color, i.e., shape (H, W, 3)

    rows = np.repeat(np.arange(H), W)
    cols = np.tile(np.arange(W), H)

    R = img[..., 0].ravel()
    G = img[..., 1].ravel()
    B = img[..., 2].ravel()

    X = np.column_stack([rows, cols, R, G, B])  # (N,5) where N = H*W
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    # here I normalize each feature dimension to [0,1] to balance spatial and color scales
    Xn = (X - mins) / (maxs - mins + 1e-9)

    return Xn, H, W


# ==============================
# 3) Here I fit a GMM and do model selection over K using 5-fold CV
# ==============================
def gmm_segment(img, K_range=range(2, 11)):
    Xn, H, W = build_features(img)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    mean_ll = []
    print("  Selecting best K...")

    # for each candidate K I compute the mean validation log-likelihood
    for K in K_range:
        fold_ll = []
        for tr_idx, va_idx in kf.split(Xn):
            gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=0)
            gmm.fit(Xn[tr_idx])
            fold_ll.append(gmm.score(Xn[va_idx]))
        mean_ll.append(np.mean(fold_ll))
        print(f"    K={K}: mean log-likelihood = {mean_ll[-1]:.4f}")

    # here I pick the K that gives the highest mean validation log-likelihood
    best_K = K_range[np.argmax(mean_ll)]
    print(f"  BEST K = {best_K}")

    # now I fit the final GMM on all pixels using the best K
    gmm = GaussianMixture(n_components=best_K, covariance_type="full", random_state=0)
    gmm.fit(Xn)

    # I take the most likely component label for each pixel and reshape into an image
    labels = gmm.predict(Xn).reshape(H, W)
    seg_norm = (labels - labels.min()) / (labels.max() - labels.min())
    seg_gray = (seg_norm * 255).astype(np.uint8)

    return best_K, mean_ll, seg_gray


# ==============================
# 4) Here I collect all image files and run the GMM segmentation on each of them
# ==============================
all_image_paths = []
for d in IMAGE_DIRS:
    if not os.path.isdir(d):
        print(f"WARNING: directory does not exist: {d}")
        continue
    for f in os.listdir(d):
        if f.lower().endswith(".jpg"):
            all_image_paths.append(os.path.join(d, f))

# here I optionally limit how many images I want to run to keep things manageable
all_image_paths = sorted(all_image_paths)
if MAX_IMAGES is not None:
    all_image_paths = all_image_paths[:MAX_IMAGES]

print(f"Found {len(all_image_paths)} images to process.")

for img_path in all_image_paths:
    fname = os.path.basename(img_path)
    print("\n====================================")
    print("Processing:", fname)

    img = io.imread(img_path)

    # if any image is grayscale, I convert it to 3 channels so my feature code still works
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # here I optionally downsample large images so the GMM runs faster
    if img.shape[0] * img.shape[1] > 300 * 300:
        img = transform.resize(img, (300, 300), anti_aliasing=True)
        img = (img * 255).astype(np.uint8)

    best_K, mean_ll, seg = gmm_segment(img)

    # here I visualize the model selection curve, the original image, and the segmentation
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    K_range = range(2, 11)
    ax[0].plot(list(K_range), mean_ll, marker="o")
    ax[0].set_title("GMM Model Selection")
    ax[0].set_xlabel("Number of components K")
    ax[0].set_ylabel("Mean log-likelihood")

    ax[1].imshow(img)
    ax[1].set_title("Original")
    ax[1].axis("off")

    ax[2].imshow(seg, cmap="gray")
    ax[2].set_title(f"GMM Segmentation (K={best_K})")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()
