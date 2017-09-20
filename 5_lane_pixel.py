# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import thresholding, plot_images, Binarizer, image_closing
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


def run(image):
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    image = corrector.run(image)
    edges = cv2.Canny(img,50,200)
    image = region_of_interest(image)

    binary_img = Binarizer.intensity(image, (78, 255))
    binary_img = image_closing(binary_img)
    
    combined = np.zeros_like(image)
    combined[:,:,0] += edges
    combined[:,:,2] += binary_img
    combined = region_of_interest(combined)
    
    return binary_img, edges, combined
    

if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
    
    import glob
    files = glob.glob('test_images//*.jpg')
    for filename in files[3:]:
        img = plt.imread(filename)
        
        binary_img, edges, combined = run(img)
        
        plot_images([img, binary_img, edges, combined],
                    ["original : {}".format(filename), "binary", "edges", "combined"])
    
    
    
    

