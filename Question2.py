import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import label2rgb
from sklearn.mixture import GaussianMixture

# ----------------------------------------------------------
# Build 5D feature vector: (row, col, R, G, B) all in [0,1]
# ----------------------------------------------------------
def build_features(img, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    H, W, _ = img.shape

    # normalize RGB to [0,1]
    img_norm = img.astype(np.float64)
    img_norm -= img_norm.min()
    if img_norm.max() > 0:
        img_norm /= img_norm.max()

    # generate row/col features normalized
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rr = rr.astype(np.float64) / (H - 1)
    cc = cc.astype(np.float64) / (W - 1)

    X = np.stack([
        rr.ravel(),
        cc.ravel(),
        img_norm[:,:,0].ravel(),
        img_norm[:,:,1].ravel(),
        img_norm[:,:,2].ravel()
    ], axis=1)

    return X, (H, W)

# ----------------------------------------------------------
# Fit GMM and return 2D label image + model
# ----------------------------------------------------------
def gmm_segment(X, img_shape, K, random_state=0):
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        max_iter=200,
        random_state=random_state
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    H, W = img_shape
    return labels.reshape(H, W), gmm

# ----------------------------------------------------------
# Process each image
# ----------------------------------------------------------
def process_image(image_path, random_state=0):
    print(f"\nProcessing: {image_path}")
    
    rng = np.random.default_rng(random_state)

    img = io.imread(image_path)
    if img.ndim == 2:  # grayscale -> RGB
        img = np.dstack([img, img, img])

    img = img.astype(np.float64)

    X, img_shape = build_features(img, rng=rng)

    # ----------- K sweep for line plot (model selection curve) -----------
    Ks = range(2, 21)
    log_liks = []

    for K in Ks:
        labelsK, gmmK = gmm_segment(X, img_shape, K, random_state + K)
        log_liks.append(gmmK.score(X))
        print(f"K={K}: Log-likelihood = {gmmK.score(X):.4f}")

    # -------- Plot Model Selection Curve --------
    plt.figure(figsize=(6,4))
    plt.plot(Ks, log_liks, marker='o')
    plt.title("Model Selection Curve (Mean Log-Likelihood vs K)")
    plt.xlabel("Number of GMM Components (K)")
    plt.ylabel("Mean Log-Likelihood")
    plt.grid(True)
    plt.show()

    # --- K = 10 segmentation
    labels10, gmm10 = gmm_segment(X, img_shape, 10, random_state)
    seg10_rgb = label2rgb(labels10, image=img.astype(np.uint8), kind='avg')

    # --- K = 20 segmentation
    labels20, gmm20 = gmm_segment(X, img_shape, 20, random_state + 1)
    seg20_rgb = label2rgb(labels20, image=img.astype(np.uint8), kind='avg')

    # -------- Plot colored segmentations --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(seg10_rgb)
    axes[1].set_title("GMM Segmentation (K=10)")
    axes[1].axis("off")

    axes[2].imshow(seg20_rgb)
    axes[2].set_title("GMM Segmentation (K=20)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return labels10, labels20


# ----------------------------------------------------------
# Automatically find BSDS images and run everything
# ----------------------------------------------------------
if __name__ == "__main__":
    
    # CHANGE THIS >>>>
    BASE_DIR = r"C:\Users\91635\Downloads\snerf_project\BSDS300\images"
    
    image_paths = []
    for sub in ["train", "test"]:
        d = os.path.join(BASE_DIR, sub)
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower().endswith((".jpg", ".png", ".bmp", ".jpeg")):
                    image_paths.append(os.path.join(d, fname))
    
    image_paths = sorted(image_paths)
    print("Found", len(image_paths), "images.")

    # Run only on 3 images for report clarity (or remove slicing)
    image_paths = image_paths[:3]

    for path in image_paths:
        process_image(path)
