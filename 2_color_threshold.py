# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import Binarizer, plot_images, image_opening

if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
    img = plt.imread('test_images/straight_lines1.jpg')
    img = corrector.run(img)
    
    intensity_bin = Binarizer.intensity(img, (112, 255))
    gx_bin = Binarizer.gradient_x(img, (10, 255))
    grad_dir_bin = Binarizer.gradient_direction(img, (0.7, 1.3))

    output = np.zeros_like(gx_bin)
    output[(gx_bin == 1) & (grad_dir_bin == 1) | (intensity_bin == 1)] = 1
    opened = image_opening(output)
     
    plot_images([img, output, opened],
                ["original", "combined", "opening"])
