import numpy as np
from pathlib import Path
from skimage import io as sk
from scipy import signal
from matplotlib import pyplot


def vijay_harris_corner_detector(input_image, k):
    size1 = np.shape(input_image)
    window = np.ones([3, 3])
    sobel_x = np.reshape([1, 0, -1, 2, 0, -2, 1, 0, -1], [3, 3])
    sobel_y = np.reshape([1, 2, 1, 0, 0, 0, -1, -2, -1], [3, 3])

    ix = signal.convolve2d(input_image, sobel_x, boundary='symm', mode='same')
    iy = signal.convolve2d(input_image, sobel_y, boundary='symm', mode='same')

    m11 = signal.convolve2d(ix * ix, window, boundary='symm', mode='same')
    m12 = signal.convolve2d(ix * iy, window, boundary='symm', mode='same')
    m22 = signal.convolve2d(iy * iy, window, boundary='symm', mode='same')

    m1 = (m11 * m22) - (m12 ** 2) - k * (m11 + m22)
    thresh = np.max(m1)/3
    m1[m1 < thresh] = 0
    return m1
