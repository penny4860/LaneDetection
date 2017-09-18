# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import thresholding, plot_images
from detector.warp import PerspectTrans
import cv2
import numpy as np
import math


    
def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    ylength, xlength = img.shape[:2]
    
    vertices = np.array([[(0, ylength),
                          (xlength/2-ylength/10, ylength*0.5),
                          (xlength/2+ylength/10, ylength*0.5),
                          (xlength, ylength)]], dtype=np.int32)

    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    
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
    maxLineGap = 25
    lines = cv2.HoughLinesP(binary,
                            10,
                            np.pi/180,
                            25,
                            np.array([]),
                            minLineLength=minLineLength,
                            maxLineGap=maxLineGap)
    
    output = np.zeros_like(img)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = _get_angle(x1, y1, x2, y2)
        
        if _which_side(x1, y1, x2, y2, binary.shape[1]) == "left":
            if theta <= -30 and theta >= -90:
                cv2.line(output, (x1, y1), (x2, y2), color=(255,0,0), thickness=10)
                # print("left", theta)
                
        else:
            if theta >= 30 and theta <= 90:
                cv2.line(output, (x1, y1), (x2, y2), color=(0,0,255), thickness=10)
                # print("right", theta)

    return output

if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    # 2. Thresholding
    # img = plt.imread('test_images/straight_lines1.jpg')
    img = plt.imread('test_images/test2.jpg')
    original = img.copy()
    img = corrector.run(img)
    img = region_of_interest(img)

    thd = thresholding(img)
    kernel = np.ones((5,5),np.uint8)
    dialate = cv2.dilate(thd, kernel, iterations = 2)
 
    img = hough(dialate, img)
    plot_images([original, thd, dialate, img],
                ["original", "binary", "dialate", "Hough"])
    

    
    
    
    

