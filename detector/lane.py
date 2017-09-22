# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import plot_images, Binarizer, image_closing, region_of_interest
import cv2
import numpy as np


np.set_printoptions(linewidth=500000)

class LanePixelDetector(object):
    """
    # Args
        distortion_corrector: DistortionCorrector
    """
     
    def __init__(self):
        pass
     
    def run(self, edge_map, binary_map):
        r_dist_map, l_dist_map = self._get_dist_map(binary_map, edge_map)
        lane_map = np.zeros_like(binary_map)
        
        # Todo : image의 row 위치에 따라서 r_dist_map+l_dist_map threshold 를 linear 하게 적용하자.
        lane_map[(abs(r_dist_map - l_dist_map) <= 3) & (r_dist_map > 0) & (l_dist_map > 0) & (r_dist_map+l_dist_map < 30)] = 255
        
        indices = np.where(lane_map == 255)
        for r, c in zip(indices[0], indices[1]):
            right_dist = int(r_dist_map[r, c])
            left_dist = int(l_dist_map[r, c])
            lane_map[r, c:c+right_dist] = 255
            lane_map[r, c-left_dist+1:c] = 255
        
        return lane_map

    def _get_dist_map(self, binary_image, edge_map):
        
        right_dist_map = np.zeros_like(binary_image).astype(float) - 1
        left_dist_map = np.zeros_like(binary_image).astype(float) - 1
        
        indices = np.where(binary_image != 0)
    
        for r, c in zip(indices[0], indices[1]):
            l_dist, r_dist = self._calc_edge_dist(r, c, edge_map)
            right_dist_map[r, c] = r_dist
            left_dist_map[r, c] = l_dist
        return right_dist_map, left_dist_map

    def _dist(self, array):
        indices = np.where(array != 0)[0]
        if indices.size == 0:
            dist = np.inf
        else:
            dist = indices[0]
        return dist

    def _calc_edge_dist(self, y, x, edge_map):
        """
        # Args
            binary_image : 2d array
            edge_map : 2d array
        # Returns
            right_dist : int
            left_dist : int
        """
        
        right_array = edge_map[y, x:]
        right_dist = self._dist(right_array)
            
        left_array = edge_map[y, :x+1][::-1]
        left_dist = self._dist(left_array)
        return left_dist, right_dist
    



if __name__ == "__main__":
    corrector = DistortionCorrector.from_pkl("..//dataset//distortion_corrector.pkl")

#     combined = np.zeros_like(img)
#     combined[:,:,0] += edges
#     combined[:,:,2] += binary_img
#     combined = region_of_interest(combined)

    detector = LanePixelDetector()
    # 1. Distortion Correction
    import glob
    files = glob.glob('..//test_images//*.jpg')
    for filename in files[:1]:
        img = plt.imread(filename)
        img = corrector.run(img)
        
        edges = cv2.Canny(img,50,200)
        binary_img = Binarizer.intensity(region_of_interest(img), (48, 255))
        binary_img = image_closing(binary_img)

        lane_map = detector.run(edges, binary_img)

        plot_images([img, lane_map],
                    ["original : {}".format(filename), "lane_map"])


