# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.lane import run
from detector.cal import DistortionCorrector
from detector.binary import plot_images
import cv2
import numpy as np


np.set_printoptions(linewidth=500000)


if __name__ == "__main__":
    
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
     
    import glob
    files = glob.glob('test_images//straight_lines1.jpg')
    for filename in files:
        img = plt.imread(filename)
        img, binary_img, edges, combined, lane_map = run(img)

        plot_images([img, combined, lane_map],
                    ["original : {}".format(filename), "combined", "lane_map"])


