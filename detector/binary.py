# -*- coding: utf-8 -*-

import numpy as np
import cv2


class _Binarizer(object):
    def __init__(self):
        pass
    
    def run(self, image, thresh):
        pass

    # Todo : 여러개를 사용할 수 있는 design pattern 을 찾아보자.
    def roi_mask(self, img, vertices=None):
        
        def _set_default_vertices(img):
            ylength, xlength = img.shape[:2]
            vertices = np.array([[(0, ylength),
                                  (xlength/2-ylength/10, ylength*0.5),
                                  (xlength/2+ylength/10, ylength*0.5),
                                  (xlength, ylength)]], dtype=np.int32)
            return vertices
        
        #defining a blank mask to start with
        mask = np.zeros_like(img)
        
        if vertices is None:
            vertices = _set_default_vertices(img)
        
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

    def _to_binaray(self, image, threshold=(0, 255)):
        """
        # Args
            image : 2d array
                uint8-scaled image

            threshold : tuple
        
        # Returns
            binary : 2d array
                whose intensity is 0 or 1
        """
        binary = np.zeros_like(image)
        binary[(image > threshold[0]) & (image <= threshold[1])] = 1
        return binary

    def _to_uint8_scale(self, image):
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        image = np.uint8(255*image/np.max(image))
        return image


class SchannelBin(_Binarizer):
    def run(self, image, thresh):
        """
        # Args
            image : 3d array
                RGB ordered image tensor
            thresh : tuple
                (minimun threshold, maximum threshold)
        # Return
            binary : 2d array
                Binary image
        """
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        s_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,2]
        binary = self._to_binaray(s_channel, thresh) * 255
        return binary


class GradientMagBin(_Binarizer):
    def run(self, image, thresh=(0, 255)):
        
        sobel_kernel=3
        # 1) Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
        # 3) Calculate the magnitude 
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = self._to_uint8_scale(sobel)

        binary = self._to_binaray(sobel, thresh)
        return binary

class GradientDirBin(_Binarizer):
    def run(self, img, thresh=(0, np.pi/2)):
        
        sobel_kernel=3
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobelx = np.absolute(sobelx)

        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobely = np.absolute(sobely)

        sobel = np.arctan2(sobely, sobelx)
        binary = self._to_binaray(sobel, thresh)
        return binary


class GxBin(_Binarizer):
    def run(self, img, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobel = np.absolute(sobel)
        sobel = self._to_uint8_scale(sobel)
        binary = self._to_binaray(sobel, thresh)
        return binary

class GyBin(_Binarizer):
    def run(self, img, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5),0)
        
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobel = np.absolute(sobel)
        sobel = self._to_uint8_scale(sobel)
        binary = self._to_binaray(sobel, thresh)
        return binary

