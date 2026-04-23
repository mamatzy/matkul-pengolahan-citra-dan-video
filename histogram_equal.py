import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def MenghitungHistogram(ImGray):
    Histogram = np.zeros((256,1), np.int32)
    for i in range(0, ImGray.shape[0]):
        for j in range(0, ImGray.shape[1]):
            r = ImGray[i,j]
            Histogram[r]=Histogram[r]+1
    return Histogram

def HistogramEqualization(ImGray):
    M, N = ImGray.shape          # dimensi citra
    L = 256                      # jumlah level intensitas
    MN = M * N                   # total piksel

    # Langkah 1: Hitung histogram
    hist = MenghitungHistogram(ImGray)

    # Langkah 2: Hitung CDF (Cumulative Distribution Function)
    cdf = np.zeros(256, np.float64)
    cdf[0] = hist[0, 0]
    for k in range(1, 256):
        cdf[k] = cdf[k - 1] + hist[k, 0]

    # Langkah 3: Hitung fungsi transformasi T(r)
    # T(r_k) = round( (L-1) / MN * CDF(r_k) )
    T = np.zeros(256, np.uint8)
    for k in range(256):
        T[k] = np.uint8(np.round((L - 1) / MN * cdf[k]))

    # Langkah 4: Terapkan transformasi ke setiap piksel
    ImOutput = np.zeros_like(ImGray)
    for i in range(M):
        for j in range(N):
            ImOutput[i, j] = T[ImGray[i, j]]

    return ImOutput, T


cam = cv . VideoCapture (0)
if not cam . isOpened () :
    print (" error opening camera ")
    exit ()

# Declare window size
cv.namedWindow('frame_mamat', cv.WINDOW_NORMAL)
cv.resizeWindow('frame_mamat', 1920, 1080)

while True :
    # Capture frame -by - frame
    ret , frame = cam . read ()
    # if frame is read correctly ret is True
    if not ret :
        print (" error in retrieving frame ")   
        break

    # (b,c,d) = frame.shape

    # for i in range (b) :
    #     for j in range (c) :
    #         frame[i,j,1] = 0 # Merubah nilai pixel pada channel merah menjadi 0
    #         # frame[i,j,1] = 0 # Merubah nilai pixel pada channel hijau
    
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame, 'POV : my laptop gweh ', (10, 30), font, 1, (255, 255, 255), 2, cv.LINE_AA)

    gray = cv . cvtColor ( frame , cv. COLOR_BGR2GRAY )
    
    edges = cv.Canny(gray, 100, 200) #100 adalah treshold bawah, dan 200 adalah treshold atas

    neg = 255 - frame

    imeq, T = HistogramEqualization(edges)

    cv . imshow ('frame_mamat', edges)

    # histogram = MenghitungHistogram(imeq)
    # plt.plot(histogram)
    # plt.show()

    if cv . waitKey (1) == ord('q') :
        break
cam . release ()
cv . destroyAllWindows ()