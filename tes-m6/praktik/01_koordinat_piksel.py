"""
Praktik 1: Sistem Koordinat Piksel
- Matrix indexing: image[row, col] = image[y, x]
- Origin di kiri atas, row bertambah ke bawah
"""
import cv2
import numpy as np

img = cv2.imread('image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("=== Ukuran Citra ===")
print(f"Shape: {gray.shape}")  # (rows, cols) = (height, width)
print(f"Height (rows): {gray.shape[0]}")
print(f"Width  (cols): {gray.shape[1]}")

# Akses piksel: image[row, col]
print("\n=== Akses Piksel ===")
print(f"Piksel [0, 0] (kiri atas)     = {gray[0, 0]}")
print(f"Piksel [0, -1] (kanan atas)   = {gray[0, -1]}")
print(f"Piksel [-1, 0] (kiri bawah)   = {gray[-1, 0]}")
print(f"Piksel [-1, -1] (kanan bawah) = {gray[-1, -1]}")

# Lihat neighborhood 5x5 di sekitar piksel [100, 100]
r, c = 100, 100
print(f"\n=== Neighborhood 5x5 di sekitar [{r}, {c}] ===")
neighborhood = gray[r-2:r+3, c-2:c+3]
print(neighborhood)

# 4-connectivity vs 8-connectivity
print(f"\n=== Ketetanggaan piksel [{r}, {c}] ===")
print(f"Nilai pusat: {gray[r, c]}")
print(f"4-connectivity (atas, bawah, kiri, kanan):")
print(f"  Atas  [{r-1},{c}] = {gray[r-1, c]}")
print(f"  Bawah [{r+1},{c}] = {gray[r+1, c]}")
print(f"  Kiri  [{r},{c-1}] = {gray[r, c-1]}")
print(f"  Kanan [{r},{c+1}] = {gray[r, c+1]}")
print(f"8-connectivity (tambah diagonal):")
print(f"  Kiri-atas  [{r-1},{c-1}] = {gray[r-1, c-1]}")
print(f"  Kanan-atas [{r-1},{c+1}] = {gray[r-1, c+1]}")
print(f"  Kiri-bawah [{r+1},{c-1}] = {gray[r+1, c-1]}")
print(f"  Kanan-bawah[{r+1},{c+1}] = {gray[r+1, c+1]}")
