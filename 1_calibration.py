# -*- coding: utf-8 -*-

import cv2
import glob
import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector


if __name__ == "__main__":
    
    # 1. Get images in gray scale
    corrector = DistortionCorrector.from_pkl("distortion_corrector.pkl")
    
    # 3. Run correction & visualize the result
    # img = cv2.imread('camera_cal/calibration1.jpg')
    img = plt.imread("test_images/straight_lines1.jpg")

    dst = corrector.run(img)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()

