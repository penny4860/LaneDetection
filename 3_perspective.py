# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.warp import PerspectTrans
import cv2
from detector.imutils import plot_images
from detector.binary import SchannelBin

if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    # 2. Thresholding
    img = plt.imread('test_images/straight_lines1.jpg')
    img = corrector.run(img)
    
    binarizer = SchannelBin()
    binary_img = binarizer.run(img, (48, 255))
    binary_img = binarizer.roi_mask(binary_img)
    
    trans = PerspectTrans(binary_img.shape[:2][::-1])
    warped = trans.run(img)

#     cv2.circle(img, center=(250, 700), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(1075, 700), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(600, 450), radius=5, thickness=10, color=(0,0,255))
#     cv2.circle(img, center=(685, 450), radius=5, thickness=10, color=(0,0,255))

    plot_images([img, binary_img, warped],
                ["original", "thresholded", "warped"])


