"""
Praktik 9: Unsharp Masking & High-Boost Filtering
- Mask = Original - Blurred
- Unsharp:   Sharpened = Original + Mask         (k=1)
- High-Boost: Sharpened = Original + k * Mask    (k>1)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../test_image_lena_ori.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

# ===== Step by step =====
# Step 1: Blur (lowpass)
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)

# Step 2: Mask = Original - Blurred (highpass / detail)
mask = img - blurred

# Step 3: Sharpened = Original + k * Mask
k_values = [0.5, 1.0, 1.5, 2.0, 3.0]

print("=== Langkah Unsharp Masking ===")
print(f"Original [100,100]  = {img[100,100]:.0f}")
print(f"Blurred  [100,100]  = {blurred[100,100]:.1f}")
print(f"Mask     [100,100]  = {mask[100,100]:.1f}")
print()
for k in k_values:
    val = img[100,100] + k * mask[100,100]
    print(f"k={k:.1f}: Sharpened = {img[100,100]:.0f} + {k} × {mask[100,100]:.1f} = {val:.1f}")

print("\nk=1  → Unsharp Masking (standar)")
print("k>1  → High-Boost Filtering (lebih tajam)")
print("k<1  → Sharpening lebih halus")

# ===== Visualisasi =====
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Pipeline
axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title('Original')
axes[0, 1].imshow(blurred, cmap='gray'); axes[0, 1].set_title('Blurred (Lowpass)')
axes[0, 2].imshow(mask + 128, cmap='gray'); axes[0, 2].set_title('Mask (Highpass) +128')
sharp_1 = np.clip(img + mask, 0, 255).astype(np.uint8)
axes[0, 3].imshow(sharp_1, cmap='gray'); axes[0, 3].set_title('Sharpened (k=1)')

# Row 2: Berbagai k
for i, k in enumerate([0.5, 1.0, 2.0, 3.0]):
    result = np.clip(img + k * mask, 0, 255).astype(np.uint8)
    axes[1, i].imshow(result, cmap='gray')
    axes[1, i].set_title(f'k = {k}')

for ax in axes.ravel():
    ax.axis('off')

plt.suptitle('Unsharp Masking & High-Boost Filtering', fontsize=14)
plt.tight_layout()
plt.show()
