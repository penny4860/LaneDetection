# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import thresholding, plot_images
from detector.warp import PerspectTrans
import cv2


if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    # 2. Thresholding
    img = plt.imread('test_images/straight_lines1.jpg')
    img = corrector.run(img)
    thd = thresholding(img, False)
    denoised = thresholding(img)
    
    trans = PerspectTrans(denoised.shape[:2][::-1])
    warped = trans.run(img)
    
    trans.to_pkl("perspective_trans.pkl")

#     cv2.circle(img, center=(250, 700), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(1075, 700), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(600, 450), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(685, 450), radius=5, thickness=10, color=(0,0,255))

    plot_images([img, denoised, warped],
                ["original", "thresholded + opening", "warped"])


