# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

def opening(image, ksize=(5,5)):
    """
    # Args
        image : 2d array
            binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return denoised

def closing(image, ksize=(5,5)):
    """
    # Args
        image : 2d array
            binary image
    """
    # kernel = np.ones(ksize, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    denoised = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return denoised

def plot_images(images, titles=None):
    _, axes = plt.subplots(1, len(images), figsize=(10,10))
    
    for img, ax, text in zip(images, axes, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(text, fontsize=30)
    plt.show()
