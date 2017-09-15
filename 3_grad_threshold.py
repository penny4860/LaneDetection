# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector

def plot_images(images, titles=None):
    _, axes = plt.subplots(1, len(images), figsize=(20,10))
    
    for img, ax, text in zip(images, axes, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(text, fontsize=30)
    plt.show()

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.absolute(sobel)
    sobel = np.uint8(255*sobel/np.max(sobel))
    
    # Create a mask of 1's where the scaled gradient magnitude 
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude 
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    sobel = np.uint8(255*sobel/np.max(sobel))
    
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel > mag_thresh[0]) & (sobel < mag_thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    sobelx = np.absolute(sobelx)
    sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    sobel = np.arctan2(sobely, sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel > thresh[0]) & (sobel < thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


# 1. Distortion Correction
corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
img = plt.imread('test_images/straight_lines1.jpg')
img = corrector.run(img)

gx = abs_sobel_thresh(img, 'x', (10, 255))
gy = abs_sobel_thresh(img, 'y', (10, 255))
mag = mag_thresh(img, 3, (30, 255))
dir = dir_threshold(img, 15, (0.7, 1.3))

plot_images([img, gx, gy, mag, dir],
            ["original", "gx", "gy", "mag", "dir"])



