# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import thresholding, plot_images
from detector.warp import PerspectTrans
import cv2
import numpy as np


def hough(binary, img):
    # Hough Transform
    minLineLength = 25
    maxLineGap = 10
    lines = cv2.HoughLinesP(binary,
                            10,
                            np.pi/180,
                            100,
                            np.array([]),
                            minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color=(255,0,0), thickness=4)
    return img

if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    # 2. Thresholding
    # img = plt.imread('test_images/straight_lines1.jpg')
    img = plt.imread('test_images/test1.jpg')
    original = img.copy()
    img = corrector.run(img)
    thd = thresholding(img)

    img = hough(thd, img)
    plot_images([original, thd, img],
                ["original", "binary", "lines"])

