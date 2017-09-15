# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import thresholding, plot_images

if __name__ == "__main__":
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    img = plt.imread('test_images/straight_lines2.jpg')
    img = corrector.run(img)
    thd = thresholding(img, False)
    denoised = thresholding(img)
     
    plot_images([img, thd, denoised],
                ["original", "thresholded", "thresholded + opening"])
