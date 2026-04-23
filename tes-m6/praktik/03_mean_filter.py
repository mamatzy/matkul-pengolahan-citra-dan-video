"""
Praktik 3: Mean Filter (Box Filter)
- Setiap piksel output = rata-rata piksel dalam window
- Semakin besar kernel, semakin blur
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../test_image_lena_noisy.png', cv2.IMREAD_GRAYSCALE)

# ===== Hitung manual pada area kecil =====
patch = img[100:107, 100:107].astype(np.float64)
print("=== Patch 7x7 dari gambar ===")
print(patch.astype(int))

# Mean filter 3x3 manual di posisi [1,1]
r, c = 1, 1
window = patch[r-1:r+2, c-1:c+2]
print(f"\nWindow 3x3 di [{r},{c}]:")
print(window.astype(int))
print(f"Rata-rata = {window.mean():.2f} → dibulatkan = {round(window.mean())}")

# ===== Bandingkan ukuran kernel =====
mean_3 = cv2.blur(img, (3, 3))
mean_5 = cv2.blur(img, (5, 5))
mean_7 = cv2.blur(img, (7, 7))
mean_15 = cv2.blur(img, (15, 15))

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
titles = ['Original (noisy)', 'Mean 3x3', 'Mean 5x5', 'Mean 7x7', 'Mean 15x15']
images = [img, mean_3, mean_5, mean_7, mean_15]

for ax, im, t in zip(axes, images, titles):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
    ax.axis('off')

plt.suptitle('Efek Ukuran Kernel Mean Filter', fontsize=14)
plt.tight_layout()
plt.show()

# ===== Lihat nilai sebelum vs sesudah =====
print("\n=== Perbandingan Nilai ===")
print(f"Original [100:105, 100:105]:\n{img[100:105, 100:105]}")
print(f"\nMean 3x3 [100:105, 100:105]:\n{mean_3[100:105, 100:105]}")
print(f"\nMean 7x7 [100:105, 100:105]:\n{mean_7[100:105, 100:105]}")
print("→ Semakin besar kernel, nilai semakin 'rata' (smooth)")
