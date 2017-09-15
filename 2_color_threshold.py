# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector

def denoising(image, ksize=[3,3]):
    """
    # Args
        image : 2d array
            binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return denoised

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


# 1. Distortion Correction
corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
img = plt.imread('test_images/straight_lines1.jpg')
img = corrector.run(img)

intensity_binary = intensity_threshold(img, (112, 255))

plot_images([img, intensity_binary],
            ["original", "s-channel"])



