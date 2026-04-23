"""
Praktik 7: Gradient Operators — Edge Detection
- Roberts, Prewitt, Sobel
- Mendeteksi tepi (perubahan intensitas)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../test_image_lena_ori.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

# ===== Lihat kernel =====
print("=== Roberts ===")
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
print(f"Gx:\n{roberts_x}\nGy:\n{roberts_y}")

print("\n=== Prewitt ===")
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
print(f"Gx:\n{prewitt_x}\nGy:\n{prewitt_y}")

print("\n=== Sobel ===")
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
print(f"Gx:\n{sobel_x}\nGy:\n{sobel_y}")

# ===== Hitung manual pada patch =====
patch = img[100:105, 100:105]
print(f"\n=== Patch 5x5 ===\n{patch.astype(int)}")

window = patch[0:3, 0:3]
print(f"\nWindow 3x3 di [1,1]:\n{window.astype(int)}")
gx = np.sum(window * sobel_x)
gy = np.sum(window * sobel_y)
mag = np.sqrt(gx**2 + gy**2)
direction = np.degrees(np.arctan2(gy, gx))
print(f"Sobel Gx = {gx:.1f}")
print(f"Sobel Gy = {gy:.1f}")
print(f"Magnitude = √({gx:.1f}² + {gy:.1f}²) = {mag:.1f}")
print(f"Direction = atan2({gy:.1f}, {gx:.1f}) = {direction:.1f}°")

# ===== Terapkan ke gambar =====
# Sobel
gx_img = cv2.filter2D(img, -1, sobel_x)
gy_img = cv2.filter2D(img, -1, sobel_y)
mag_sobel = np.sqrt(gx_img**2 + gy_img**2)
mag_sobel = np.clip(mag_sobel, 0, 255).astype(np.uint8)

# Prewitt
gx_p = cv2.filter2D(img, -1, prewitt_x)
gy_p = cv2.filter2D(img, -1, prewitt_y)
mag_prewitt = np.sqrt(gx_p**2 + gy_p**2)
mag_prewitt = np.clip(mag_prewitt, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title('Original')
axes[0, 1].imshow(np.abs(gx_img), cmap='gray'); axes[0, 1].set_title('Sobel Gx (horizontal edges)')
axes[0, 2].imshow(np.abs(gy_img), cmap='gray'); axes[0, 2].set_title('Sobel Gy (vertical edges)')
axes[1, 0].imshow(mag_sobel, cmap='gray'); axes[1, 0].set_title('Sobel Magnitude')
axes[1, 1].imshow(mag_prewitt, cmap='gray'); axes[1, 1].set_title('Prewitt Magnitude')
axes[1, 2].imshow(np.abs(gx_img) + np.abs(gy_img), cmap='gray'); axes[1, 2].set_title('Sobel |Gx| + |Gy| (approx)')

for ax in axes.ravel():
    ax.axis('off')

plt.suptitle('Gradient Operators — Edge Detection', fontsize=14)
plt.tight_layout()
plt.show()
