# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import thresholding, plot_images
import cv2
import numpy as np

if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    # 2. Thresholding
    img = plt.imread('test_images/straight_lines1.jpg')
    img = corrector.run(img)
    thd = thresholding(img, False)
    denoised = thresholding(img)

#     cv2.circle(img, center=(250, 700), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(1075, 700), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(600, 450), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(685, 450), radius=5, thickness=10, color=(0,0,255))

    src = np.array([(250, 700), (1075, 700), (600, 450), (685, 450)]).astype(np.float32)
    h, w = img.shape[:2]
    x_offset = 300
    y_offset = 50
    dst = np.array([(x_offset, h-y_offset), (w-x_offset, h-y_offset), (x_offset, y_offset), (w-x_offset, y_offset)]).astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    plot_images([img, denoised, warped],
                ["original", "thresholded + opening", "warped"])


