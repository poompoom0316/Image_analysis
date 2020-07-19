import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage


# Helper function for calculating entropy
def entp(x):
    temp = np.multiply(x, np.log(x))
    temp[np.isnan(temp)] = 0
    return temp


def maxentropy(img, depth=128):
    # Maximum entropy
    H = cv2.calcHist([img], [0], None, [depth], [0, depth])
    H = H / np.sum(H)
    theta = np.zeros(depth)
    Hf = np.zeros(depth)
    Hb = np.zeros(depth)

    for T in range(1, depth - 1):
        Hf[T] = - np.sum(entp(H[:T - 1] / np.sum(H[1:T - 1])))
        Hb[T] = - np.sum(entp(H[T:] / np.sum(H[T:])))
        theta[T] = Hf[T] + Hb[T]

    theta_max = np.argmax(theta)
    img_out = img > theta_max

    return img_out


# 画像の読み込み
img = cv2.imread("Data/hela-cells.png")[:, :, 0]
maxValue = 128
adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # cv2.ADAPTIVE_THRESH_MEAN_C #cv2.ADAPTIVE_THRESH_GAUSSIAN_C
thresholdType = cv2.THRESH_BINARY  # cv2.THRESH_BINARY #cv2.THRESH_BINARY_INV
blockSize = 5  # odd number like 3,5,7,9,11
C = -3  # constant to be subtracted
im_thresholded = cv2.adaptiveThreshold(img, maxValue, adaptiveMethod, thresholdType, blockSize, C)
labelarray, particle_count = ndimage.measurements.label(im_thresholded)
ret2, th2 = cv2.threshold(img, 0, 128, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(th2)

threshold = maxentropy(img)
thresholded = img.copy()
thresholded[~threshold] = 0.
labelarray, particle_count = ndimage.measurements.label(thresholded)
plt.imshow(labelarray)

for i in range(particle_count):
    thresholded = img.copy()
    thresholded[labelarray != i] = 0.
    area = (labelarray == i).sum()

    perimeter = cv2.arcLength(thresholded, True)
    circle_level = 4.0 * np.pi * area / (perimeter * perimeter);  # perimeter = 0 のとき気をつける
