"""
Praktik 4: Gaussian Filter
- Bobot lebih besar di pusat, lebih kecil di tepi
- Parameter sigma menentukan seberapa blur
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===== Buat kernel Gaussian manual =====
def buat_kernel_gaussian(size, sigma):
    half = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x, y = j - half, i - half
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # normalisasi agar total = 1
    return kernel

# Bandingkan sigma
print("=== Kernel Gaussian 5x5, sigma=0.5 ===")
k1 = buat_kernel_gaussian(5, 0.5)
print(np.round(k1, 4))
print(f"Sum = {k1.sum():.4f}")

print("\n=== Kernel Gaussian 5x5, sigma=1.0 ===")
k2 = buat_kernel_gaussian(5, 1.0)
print(np.round(k2, 4))
print(f"Sum = {k2.sum():.4f}")

print("\n=== Kernel Gaussian 5x5, sigma=2.0 ===")
k3 = buat_kernel_gaussian(5, 2.0)
print(np.round(k3, 4))
print(f"Sum = {k3.sum():.4f}")

print("\nSigma kecil → bobot terpusat di tengah (blur sedikit)")
print("Sigma besar → bobot lebih merata (blur banyak)")

# ===== Terapkan ke gambar =====
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

gauss_05 = cv2.GaussianBlur(img, (5, 5), 0.5)
gauss_10 = cv2.GaussianBlur(img, (5, 5), 1.0)
gauss_20 = cv2.GaussianBlur(img, (5, 5), 2.0)
gauss_big = cv2.GaussianBlur(img, (15, 15), 3.0)

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
titles = ['Original', 'σ=0.5', 'σ=1.0', 'σ=2.0', 'σ=3.0 (15x15)']
images = [img, gauss_05, gauss_10, gauss_20, gauss_big]

for ax, im, t in zip(axes, images, titles):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
    ax.axis('off')

plt.suptitle('Efek Sigma pada Gaussian Filter', fontsize=14)
plt.tight_layout()
plt.show()
