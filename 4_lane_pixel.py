# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.lane import LanePixelDetector
from detector.cal import DistortionCorrector
from detector.binary import plot_images
import cv2
import numpy as np

from detector.binary import Binarizer, region_of_interest, image_closing

np.set_printoptions(linewidth=500000)


if __name__ == "__main__":
    
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
    detector = LanePixelDetector()
     
    import glob
    files = glob.glob('test_images//straight_lines1.jpg')
    for filename in files:
        img = plt.imread(filename)
        img = corrector.run(img)
        
        edges = cv2.Canny(img,50,200)
        binary_img = Binarizer.intensity(region_of_interest(img), (48, 255))
        binary_img = image_closing(binary_img)

        lane_map = detector.run(edges, binary_img)

        plot_images([img, lane_map],
                    ["original : {}".format(filename), "lane_map"])

