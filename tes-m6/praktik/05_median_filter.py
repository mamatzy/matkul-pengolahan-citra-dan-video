"""
Praktik 5: Median Filter
- Output = nilai median (tengah) dari window
- Sangat efektif untuk salt-and-pepper noise
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../test_image_lena_noisy.png', cv2.IMREAD_GRAYSCALE)

# ===== Hitung median manual =====
patch = img[100:107, 100:107].copy()
print("=== Patch 7x7 ===")
print(patch)

r, c = 1, 1
window = patch[r-1:r+2, c-1:c+2].flatten()
sorted_vals = np.sort(window)
print(f"\nWindow 3x3 di [{r},{c}]: {window}")
print(f"Sorted: {sorted_vals}")
print(f"Median (posisi tengah): {sorted_vals[4]}")
print("→ Outlier (0 atau 255) otomatis tereliminasi!")

# ===== Bandingkan Mean vs Median =====
mean_3 = cv2.blur(img, (3, 3))
median_3 = cv2.medianBlur(img, 3)
median_5 = cv2.medianBlur(img, 5)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ['Original (noisy)', 'Mean 3x3', 'Median 3x3', 'Median 5x5']
images = [img, mean_3, median_3, median_5]

for ax, im, t in zip(axes, images, titles):
    ax.imshow(im, cmap='gray')
    ax.set_title(t)
    ax.axis('off')

plt.suptitle('Mean Filter vs Median Filter pada Salt-and-Pepper Noise', fontsize=14)
plt.tight_layout()
plt.show()

# ===== Perbandingan nilai =====
print("\n=== Perbandingan Nilai [100:105, 100:105] ===")
print(f"Original:\n{img[100:105, 100:105]}")
print(f"\nMean 3x3:\n{mean_3[100:105, 100:105]}")
print(f"\nMedian 3x3:\n{median_3[100:105, 100:105]}")
print("\n→ Median lebih baik mempertahankan tepi (edge)")
print("→ Mean menghasilkan nilai 'abu-abu' di sekitar noise")
