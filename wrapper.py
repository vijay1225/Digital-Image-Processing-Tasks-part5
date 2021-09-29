import numpy as np
from pathlib import Path
from skimage import io as sk
from scipy import signal
from matplotlib import pyplot
from skimage import transform
from allfunctions import *


def problem1_a():
    img1_path = Path('Data/grey52.png')
    img1 = sk.imread(img1_path)
    img2_path = Path('Data/screenshot.png')
    img2 = sk.imread(img2_path)

    output1 = vijay_harris_corner_detector(img1, k=0.04)
    output2 = vijay_harris_corner_detector(img2, k=0.04)
    pyplot.subplot(221)
    pyplot.imshow(img1, cmap='gray')
    pyplot.title('Original')
    pyplot.subplot(222)
    pyplot.imshow(output1, cmap='gray')
    pyplot.title('detected corners')
    pyplot.subplot(223)
    pyplot.imshow(img2, cmap='gray')
    pyplot.title('Original')
    pyplot.subplot(224)
    pyplot.imshow(output2, cmap='gray')
    pyplot.title('detected corners')
    pyplot.show()


def problem1_b():
    img1_path = Path('Data/screenshot.png')
    img1 = sk.imread(img1_path)
    img2_path = Path('Data/chess.png')
    img2 = sk.imread(img2_path)
    size1 = np.shape(img1)
    size2 = np.shape(img2)
    scaling_factor = 2
    scale1 = transform.resize(img1, (int(size1[0] * scaling_factor), int(size1[1] * scaling_factor)))
    scale2 = transform.resize(img2, (int(size2[0] * scaling_factor), int(size2[1] * scaling_factor)))

    rotate1 = transform.rotate(img1, 45)
    rotate2 = transform.rotate(img2, 45)

    noise_add1 = img1 + np.random.normal(4, 30, size1)
    noise_add2 = img2 + np.random.normal(4, 30, size2)

    output11 = vijay_harris_corner_detector(img1, k=0.04)
    output21 = vijay_harris_corner_detector(img2, k=0.04)
    output12 = vijay_harris_corner_detector(scale1, k=0.04)
    output22 = vijay_harris_corner_detector(scale2, k=0.04)
    output13 = vijay_harris_corner_detector(rotate1, k=0.04)
    output23 = vijay_harris_corner_detector(rotate2, k=0.04)
    output14 = vijay_harris_corner_detector(noise_add1, k=0.04)
    output24 = vijay_harris_corner_detector(noise_add2, k=0.04)
    pyplot.figure(1)
    pyplot.subplot(241)
    pyplot.imshow(img1, cmap='gray')
    pyplot.title('Original')
    pyplot.subplot(242)
    pyplot.imshow(scale1, cmap='gray')
    pyplot.title('scaled version')
    pyplot.subplot(243)
    pyplot.imshow(rotate1, cmap='gray')
    pyplot.title('rotated version')
    pyplot.subplot(244)
    pyplot.imshow(noise_add1, cmap='gray')
    pyplot.title('noise added')
    pyplot.subplot(245)
    pyplot.imshow(output11, cmap='gray')
    pyplot.title('corners')
    pyplot.subplot(246)
    pyplot.imshow(output12, cmap='gray')
    pyplot.title('corners')
    pyplot.subplot(247)
    pyplot.imshow(output13, cmap='gray')
    pyplot.title('corners')
    pyplot.subplot(248)
    pyplot.imshow(output14, cmap='gray')
    pyplot.title('corners')

    pyplot.figure(2)
    pyplot.subplot(241)
    pyplot.imshow(img2, cmap='gray')
    pyplot.title('Original')
    pyplot.subplot(242)
    pyplot.imshow(scale2, cmap='gray')
    pyplot.title('scaled version')
    pyplot.subplot(243)
    pyplot.imshow(rotate2, cmap='gray')
    pyplot.title('rotated version')
    pyplot.subplot(244)
    pyplot.imshow(noise_add2, cmap='gray')
    pyplot.title('noise added')
    pyplot.subplot(245)
    pyplot.imshow(output21, cmap='gray')
    pyplot.title('corners')
    pyplot.subplot(246)
    pyplot.imshow(output22, cmap='gray')
    pyplot.title('corners')
    pyplot.subplot(247)
    pyplot.imshow(output23, cmap='gray')
    pyplot.title('corners')
    pyplot.subplot(248)
    pyplot.imshow(output24, cmap='gray')
    pyplot.title('corners')
    pyplot.show()


if __name__ == '__main__':
    # problem1_a()
    problem1_b()
