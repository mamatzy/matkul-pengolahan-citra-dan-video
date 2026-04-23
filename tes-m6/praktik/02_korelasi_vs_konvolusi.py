"""
Praktik 2: Korelasi vs Konvolusi
- Korelasi: kernel langsung di-overlay
- Konvolusi: kernel di-flip 180° dulu
- Untuk kernel simetris, hasilnya SAMA
"""
import cv2
import numpy as np

# ===== Demo pada matriks kecil =====
image = np.array([
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55],
    [20, 30, 40, 50, 60],
    [25, 35, 45, 55, 65],
    [30, 40, 50, 60, 70]
], dtype=np.float64)

kernel_asimetris = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=np.float64)

kernel_flipped = kernel_asimetris[::-1, ::-1]  # flip 180°

print("=== Kernel Asli ===")
print(kernel_asimetris)
print("\n=== Kernel Flipped 180° ===")
print(kernel_flipped)

# Hitung manual di posisi [1,1]
r, c = 1, 1
window = image[r-1:r+2, c-1:c+2]
print(f"\n=== Window 3x3 di posisi [{r},{c}] ===")
print(window)

korelasi_val = np.sum(window * kernel_asimetris)
konvolusi_val = np.sum(window * kernel_flipped)
print(f"\nKorelasi  (kernel asli):    {window.ravel()} . {kernel_asimetris.ravel()} = {korelasi_val}")
print(f"Konvolusi (kernel flipped): {window.ravel()} . {kernel_flipped.ravel()} = {konvolusi_val}")
print(f"Hasilnya BEDA karena kernel tidak simetris!")

# ===== Bandingkan dengan kernel simetris (mean) =====
print("\n" + "="*50)
kernel_mean = np.ones((3, 3), dtype=np.float64) / 9
kernel_mean_flipped = kernel_mean[::-1, ::-1]

print("=== Kernel Mean (simetris) ===")
print(kernel_mean)
print("\n=== Kernel Mean Flipped ===")
print(kernel_mean_flipped)

korelasi_mean = np.sum(window * kernel_mean)
konvolusi_mean = np.sum(window * kernel_mean_flipped)
print(f"\nKorelasi  = {korelasi_mean:.4f}")
print(f"Konvolusi = {konvolusi_mean:.4f}")
print(f"Hasilnya SAMA karena kernel simetris!")

# ===== Pada gambar nyata =====
print("\n" + "="*50)
print("=== Pada Gambar Lena ===")
img = cv2.imread('../test_image_lena_ori.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)

# cv2.filter2D melakukan KORELASI (bukan konvolusi)
result_corr = cv2.filter2D(img, -1, kernel_asimetris)
result_conv = cv2.filter2D(img, -1, kernel_flipped)  # manual flip = konvolusi

print(f"Hasil korelasi [100,100]  = {result_corr[100,100]}")
print(f"Hasil konvolusi [100,100] = {result_conv[100,100]}")
print("cv2.filter2D = korelasi, bukan konvolusi!")
