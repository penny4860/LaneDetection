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


def intensity_threshold(image, thresh):
    """
    # Args
        image : 3d array
            RGB ordered image tensor
        thresh : tuple
            (minimun threshold, maximum threshold)
    """
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary


def gradient_threshold(image, thresh):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
     
    # Threshold x gradient
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary


# 1. Distortion Correction
corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
img = plt.imread('test_images/straight_lines1.jpg')
img = corrector.run(img)

intensity_binary = intensity_threshold(img, (128, 255))
grad_binary = gradient_threshold(img, (10, 255))

combined_binary = np.zeros_like(intensity_binary)
combined_binary[(intensity_binary == 1) | (grad_binary == 1)] = 1


plot_images([img, intensity_binary, grad_binary, combined_binary],
            ["original", "s-channel", "grad", "combined"])



