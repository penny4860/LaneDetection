# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import thresholding, plot_images, Binarizer, image_closing, region_of_interest
from detector.warp import PerspectTrans
import cv2
import numpy as np
import math


def run(img):
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    img = corrector.run(img)
    edges = cv2.Canny(img,50,200)
    img = region_of_interest(img)

    binary_img = Binarizer.intensity(img, (78, 255))
    binary_img = image_closing(binary_img)
    
    combined = np.zeros_like(img)
    combined[:,:,0] += edges
    combined[:,:,2] += binary_img
    combined = region_of_interest(combined)
    
    return img, binary_img, edges, combined


if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
    
    import glob
    files = glob.glob('test_images//*.jpg')
    for filename in files[3:]:
        img = plt.imread(filename)
        img, binary_img, edges, combined = run(img)
        
        plot_images([img, binary_img, edges, combined],
                    ["original : {}".format(filename), "binary", "edges", "combined"])
    
    
    
    

