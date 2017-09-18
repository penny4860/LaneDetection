# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Todo : 구조 정리
def thresholding(image, do_opening=False, do_closing=True):
    intensity_bin = Binarizer.intensity(image, (96, 255))
    gx_bin = Binarizer.gradient_x(image, (5, 255))

    output = np.zeros_like(gx_bin)
    output[(gx_bin == 1) & (intensity_bin == 1)] = 255
    if do_opening:
        output = image_opening(output)
    if do_closing:
        output = image_closing(output)
    return output


def _to_uint8_scale(image):
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    image = np.uint8(255*image/np.max(image))
    return image


def _to_binaray(image, threshold=(0, 255)):
    binary = np.zeros_like(image)
    binary[(image > threshold[0]) & (image <= threshold[1])] = 1
    return binary


class Binarizer(object):
    

    @staticmethod
    def intensity(image, thresh):
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
        binary = _to_binaray(s_channel, thresh)
        return binary

    @staticmethod
    def gradient_x(img, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobel = np.absolute(sobel)
        sobel = _to_uint8_scale(sobel)
        
        binary = _to_binaray(sobel, thresh)
        return binary

    @staticmethod
    def gradient_y(img, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobel = np.absolute(sobel)
        sobel = _to_uint8_scale(sobel)

        binary = _to_binaray(sobel, thresh)
        return binary

    @staticmethod
    def gradient_magnitude(img, thresh=(0, 255), sobel_kernel=3):
        
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
        # 3) Calculate the magnitude 
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = _to_uint8_scale(sobel)

        binary = _to_binaray(sobel, thresh)
        
        return binary

    @staticmethod
    def gradient_direction(img, thresh=(0, np.pi/2), sobel_kernel=3):
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobelx = np.absolute(sobelx)

        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobely = np.absolute(sobely)

        sobel = np.arctan2(sobely, sobelx)
        binary = _to_binaray(sobel, thresh)
        return binary
    
    
def image_opening(image, ksize=(3,3)):
    """
    # Args
        image : 2d array
            binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    denoised = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return denoised

def image_closing(image, ksize=(5,5)):
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
    _, axes = plt.subplots(1, len(images), figsize=(20,10))
    
    for img, ax, text in zip(images, axes, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(text, fontsize=30)
    plt.show()

