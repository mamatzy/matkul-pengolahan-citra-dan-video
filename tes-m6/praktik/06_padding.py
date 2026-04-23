"""
Praktik 6: Border Handling & Padding
- Saat kernel di tepi, perlu padding
- Mode: zero, replicate, reflect
"""
import cv2
import numpy as np

# Matriks kecil untuk melihat efek padding
image = np.array([
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55],
    [20, 30, 40, 50, 60],
    [25, 35, 45, 55, 65],
    [30, 40, 50, 60, 70]
], dtype=np.uint8)

print("=== Image Asli 5x5 ===")
print(image)

# Zero padding (tambah border 1 piksel berisi 0)
zero_pad = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
print("\n=== Zero Padding (7x7) ===")
print(zero_pad)

# Replicate padding (duplikasi piksel tepi)
rep_pad = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
print("\n=== Replicate Padding (7x7) ===")
print(rep_pad)

# Reflect padding (mirror dari dalam)
ref_pad = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
print("\n=== Reflect Padding (7x7) ===")
print(ref_pad)

# ===== Efek pada filtering =====
print("\n" + "="*50)
print("=== Efek Padding pada Mean Filter 3x3 ===")
kernel = np.ones((3, 3), dtype=np.float64) / 9

# Filter dengan zero padding
result_zero = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
print(f"\nZero padding:\n{result_zero}")

# Filter dengan replicate padding
result_rep = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
print(f"\nReplicate padding:\n{result_rep}")

# Filter dengan reflect padding
result_ref = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
print(f"\nReflect padding:\n{result_ref}")

print("\n→ Perhatikan nilai di TEPI dan SUDUT berbeda untuk setiap mode!")
print("→ Zero padding menghasilkan nilai lebih gelap di tepi")
