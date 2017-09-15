# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import Binarizer, plot_images

if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
    img = plt.imread('test_images/straight_lines1.jpg')
    img = corrector.run(img)
    
    intensity_bin = Binarizer.intensity(img, (112, 255))
    
    plot_images([img, intensity_bin],
                ["original", "s-channel"])

    gx_bin = Binarizer.gradient_x(img, (10, 255))
    gy_bin = Binarizer.gradient_y(img, (10, 255))
    grad_mag_bin = Binarizer.gradient_magnitude(img, (30, 255))
    grad_dir_bin = Binarizer.gradient_direction(img, (0.7, 1.3))

    plot_images([img, gx_bin, grad_mag_bin, grad_dir_bin],
                ["original", "gx", "mag", "thera"])

    output = np.zeros_like(gx_bin)
    output[(gx_bin == 1) & (grad_dir_bin == 1)] = 1
     
    plot_images([img, output],
                ["original", "combined"])
