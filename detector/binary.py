# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt


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
        binary = np.zeros_like(s_channel)
        binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary

    @staticmethod
    def gradient_x(img, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        sobel = np.absolute(sobel)
        sobel = np.uint8(255*sobel/np.max(sobel))
        
        # Create a mask of 1's where the scaled gradient magnitude 
        binary_output = np.zeros_like(sobel)
        binary_output[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
        return binary_output

    @staticmethod
    def gradient_y(img, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        sobel = np.absolute(sobel)
        sobel = np.uint8(255*sobel/np.max(sobel))
        
        # Create a mask of 1's where the scaled gradient magnitude 
        binary_output = np.zeros_like(sobel)
        binary_output[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
        return binary_output

    @staticmethod
    def gradient_magnitude(img, mag_thresh=(0, 255), sobel_kernel=3):
        
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

    @staticmethod
    def gradient_direction(img, thresh=(0, np.pi/2), sobel_kernel=3):
        
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

