# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import thresholding, plot_images
from detector.warp import PerspectTrans
import cv2
import numpy as np
import math

def hough(binary, img):
    
    def _get_angle(x1, y1, x2, y2):
        """Get angle to degree unit"""
        dx = x2 - x1
        dy = y2 - y1
        return np.arctan2(dy, dx)*180/math.pi
    
    def _which_side(x1, y1, x2, y2, width):
        if ((x1 + x2) / 2) < width / 2:
            return "left"
        else:
            return "right"
        
    # Hough Transform
    minLineLength = 5
    maxLineGap = 10
    lines = cv2.HoughLinesP(binary,
                            10,
                            np.pi/180,
                            25,
                            np.array([]),
                            minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = _get_angle(x1, y1, x2, y2)
        
        if _which_side(x1, y1, x2, y2, binary.shape[1]) == "left":
            if theta <= -30 and theta >= -90:
                cv2.line(img, (x1, y1), (x2, y2), color=(255,0,0), thickness=4)
                # print("left", theta)
                
        else:
            if theta >= 30 and theta <= 90:
                cv2.line(img, (x1, y1), (x2, y2), color=(0,0,255), thickness=4)
                # print("right", theta)

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

