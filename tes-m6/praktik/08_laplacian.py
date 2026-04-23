"""
Praktik 8: Laplacian Filter (Second Derivative)
- Mendeteksi tepi ke segala arah (isotropic)
- Jumlah koefisien kernel = 0
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../test_image_lena_ori.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

# ===== Kernel Laplacian =====
lap4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
lap8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)

print("=== Laplacian 4-neighbor ===")
print(lap4)
print(f"Sum koefisien: {lap4.sum()}")

print("\n=== Laplacian 8-neighbor ===")
print(lap8)
print(f"Sum koefisien: {lap8.sum()}")

# ===== Hitung manual =====
patch = img[100:105, 100:105]
print(f"\n=== Patch 5x5 ===\n{patch.astype(int)}")

window = patch[0:3, 0:3]
lap_val = np.sum(window * lap4)
print(f"\nLaplacian 4-neighbor di [1,1]: {lap_val:.1f}")
print("Nilai positif besar = tepi")

# ===== Terapkan ke gambar =====
result_4 = cv2.filter2D(img, -1, lap4)
result_8 = cv2.filter2D(img, -1, lap8)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(img, cmap='gray'); axes[0].set_title('Original')
axes[1].imshow(np.abs(result_4), cmap='gray'); axes[1].set_title('Laplacian (4-neighbor)')
axes[2].imshow(np.abs(result_8), cmap='gray'); axes[2].set_title('Laplacian (8-neighbor)')

for ax in axes:
    ax.axis('off')

plt.suptitle('Laplacian Filter — Second Derivative', fontsize=14)
plt.tight_layout()
plt.show()

# ===== Sharpening dengan Laplacian =====
# Sharpened = Original - c * Laplacian (c=1 karena center negatif)
sharpened_4 = img - result_4
sharpened_4 = np.clip(sharpened_4, 0, 255).astype(np.uint8)

sharpened_8 = img - result_8
sharpened_8 = np.clip(sharpened_8, 0, 255).astype(np.uint8)

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
axes2[0].imshow(img.astype(np.uint8), cmap='gray'); axes2[0].set_title('Original')
axes2[1].imshow(sharpened_4, cmap='gray'); axes2[1].set_title('Sharpened (Lap 4-neighbor)')
axes2[2].imshow(sharpened_8, cmap='gray'); axes2[2].set_title('Sharpened (Lap 8-neighbor)')

for ax in axes2:
    ax.axis('off')

plt.suptitle('Sharpening dengan Laplacian', fontsize=14)
plt.tight_layout()
plt.show()

print("\n=== Perbandingan Nilai [100:103, 100:103] ===")
print(f"Original:\n{img[100:103, 100:103].astype(int)}")
print(f"\nSharpened (4-neighbor):\n{sharpened_4[100:103, 100:103]}")
print("→ Detail/tepi menjadi lebih tajam")
