# -*- coding: utf-8 -*-

import numpy as np
import cv2


class _BinExtractor(object):
    """
    # Args
        threshold : tuple
            (min_threshold, max_threshold)
    """
    def __init__(self, threshold=(48, 255)):
        self._threshold = threshold
    
    def run(self, image):
        pass

    def _to_binaray(self, image):
        """
        # Args
            image : 2d array
                uint8-scaled image
        
        # Returns
            binary : 2d array
                whose intensity is 0 or 1
        """
        binary = np.zeros_like(image)
        binary[(image > self._threshold[0]) & (image <= self._threshold[1])] = 1
        return binary

    def _to_uint8_scale(self, image):
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        image = np.uint8(255*image/np.max(image))
        return image


class SchannelBin(_BinExtractor):
    def run(self, image):
        """
        # Args
            image : 3d array
                RGB ordered image tensor
        # Return
            binary : 2d array
                Binary image
        """
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        s_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,2]
        binary = self._to_binaray(s_channel) * 255
        return binary


class GradientMagBin(_BinExtractor):
    def run(self, image):
        
        sobel_kernel=3
        # 1) Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
        # 3) Calculate the magnitude 
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = self._to_uint8_scale(sobel)

        binary = self._to_binaray(sobel)
        return binary

class GradientDirBin(_BinExtractor):
    def run(self, img):
        
        sobel_kernel=3
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobelx = np.absolute(sobelx)

        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobely = np.absolute(sobely)

        sobel = np.arctan2(sobely, sobelx)
        binary = self._to_binaray(sobel)
        return binary


class GxBin(_BinExtractor):
    def run(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobel = np.absolute(sobel)
        sobel = self._to_uint8_scale(sobel)
        binary = self._to_binaray(sobel)
        return binary

class GyBin(_BinExtractor):
    def run(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobel = np.absolute(sobel)
        sobel = self._to_uint8_scale(sobel)
        binary = self._to_binaray(sobel)
        return binary


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import glob
    
    bin_extractor = SchannelBin((48, 255))
    files = glob.glob('..//test_images//*.jpg')
    for filename in files[:1]:
        img = plt.imread(filename)
        binary_img = bin_extractor.run(img)
        plt.imshow(binary_img, cmap="gray")
        plt.show()




