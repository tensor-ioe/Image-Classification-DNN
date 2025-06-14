import os
import random
import numpy as np
from PIL import Image
import torch


def compute_pca_stream(
    train_dir: str,
    max_images: int = 1000,
    pixels_per_image: int = 100000,
    chunk_size: int = 20000
):
    """
    Streams pixel samples in chunks to compute mean and covariance without
    storing all samples in memory.

    Args:
        train_dir: directory of images
        max_images: max number of images to sample
        pixels_per_image: total pixels to sample per image
        chunk_size: process pixels in chunks to bound memory use

    Returns:
        mean: (3,) array of per-channel means
        std: (3,) array of per-channel stddevs
        eigvals: (3,) PCA eigenvalues
        eigvecs: (3,3) PCA eigenvectors
    """
    # Initialize accumulators for total pixel count, RGB sum, and covariance matrix
    N_total = 0
    sum_rgb = np.zeros(3, dtype=np.float64)  # Sum of RGB values
    sum_outer = np.zeros((3, 3), dtype=np.float64)  # Sum of outer products for covariance

    # Get a list of image filenames in the directory, limited to max_images
    fnames = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ][:max_images]

    # Process each image
    for fname in fnames:
        # Open the image and convert it to RGB format
        img = Image.open(os.path.join(train_dir, fname)).convert('RGB')
        # Convert the image to a NumPy array normalized to [0, 1]
        arr_img = np.asarray(img, np.float32) / 255.0
        H, W, _ = arr_img.shape  # Get image dimensions

        remaining = pixels_per_image  # Pixels left to sample from this image
        while remaining > 0:
            # Determine the batch size for this chunk
            bs = min(chunk_size, remaining)
            # Randomly sample pixel coordinates
            ys = np.random.randint(0, H, size=bs)
            xs = np.random.randint(0, W, size=bs)
            # Extract the sampled pixel values
            samples = arr_img[ys, xs, :]  # Shape: (bs, 3)

            # Update accumulators
            N_total += bs  # Increment total pixel count
            sum_rgb += samples.sum(axis=0)  # Add RGB values
            sum_outer += samples.T @ samples  # Add outer product of samples

            remaining -= bs  # Decrease remaining pixel count

    # Compute the mean RGB values
    mean = sum_rgb / N_total
    # Compute the covariance matrix
    cov = (sum_outer - N_total * np.outer(mean, mean)) / (N_total - 1)
    # Compute the standard deviation for each channel
    std = np.sqrt(np.diag(cov))

    # Perform PCA using eigen-decomposition of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Return the computed mean, standard deviation, eigenvalues, and eigenvectors
    return mean.astype(np.float32), std.astype(np.float32), \
           eigvals.astype(np.float32), eigvecs.astype(np.float32)




# 4) Putting it all together
if __name__ == "__main__":
    train_dir = "Pistachio_Image_Dataset/Pistachio_Image_Dataset"
    save_dir = os.path.join(train_dir, "augmented")
    os.makedirs(save_dir, exist_ok=True)


    print("Computing PCA...")

    mean, std, eigvals, eigvecs = compute_pca_stream(train_dir=train_dir,max_images=2000,pixels_per_image=300000, chunk_size=10000)

    print("Mean (R,G,B):", mean)
    print("Standard deviations (R,G,B):", std)
    print("Eigenvalues:", eigvals)
    print("First principal direction:", eigvecs[:,0])

    # Save PCA values to a file using torch
    pca_data = {
        "mean": mean,
        "std": std,
        "eigvals": eigvals,
        "eigvecs": eigvecs
    }
    torch.save(pca_data, "pca_values.pth")
    print(f"PCA values saved to pca_values.pth")
