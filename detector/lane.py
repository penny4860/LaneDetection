# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.binary import plot_images, Binarizer, image_closing, region_of_interest
import cv2
import numpy as np


np.set_printoptions(linewidth=500000)


def run(img):
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")

    img = corrector.run(img)
    edges = cv2.Canny(img,50,200)
    img = region_of_interest(img)

    binary_img = Binarizer.intensity(img, (48, 255))
    binary_img = image_closing(binary_img)
    
    combined = np.zeros_like(img)
    combined[:,:,0] += edges
    combined[:,:,2] += binary_img
    combined = region_of_interest(combined)
    
    r_dist_map, l_dist_map = get_dist_map(binary_img, edges)
    lane_map = np.zeros_like(binary_img)
    lane_map[(abs(r_dist_map - l_dist_map) <= 3) & (r_dist_map > 0) & (l_dist_map > 0) & (r_dist_map+l_dist_map < 20)] = 255
    
    return img, binary_img, edges, combined, lane_map


def calc_edge_dist(y, x, edge_map):
    """
    # Args
        binary_image : 2d array
        edge_map : 2d array
    # Returns
        right_dist : int
        left_dist : int
    """
    def _dist(array):
        indices = np.where(array != 0)[0]
        if indices.size == 0:
            dist = np.inf
        else:
            dist = indices[0]
        return dist
    
    right_array = edge_map[y, x:]
    right_dist = _dist(right_array)
        
    left_array = edge_map[y, :x+1][::-1]
    left_dist = _dist(left_array)
    return left_dist, right_dist

def get_dist_map(binary_image, edge_map):
    
    right_dist_map = np.zeros_like(binary_image).astype(float) - 1
    left_dist_map = np.zeros_like(binary_image).astype(float) - 1
    
    indices = np.where(binary_image != 0)

    for r, c in zip(indices[0], indices[1]):
        l_dist, r_dist = calc_edge_dist(r, c, edge_map)
        right_dist_map[r, c] = r_dist
        left_dist_map[r, c] = l_dist
    return right_dist_map, left_dist_map


if __name__ == "__main__":
    
    # 1. Distortion Correction
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
     
    import glob
    files = glob.glob('test_images//*.jpg')
    for filename in files[5:]:
        img = plt.imread(filename)
        img, binary_img, edges, combined, lane_map = run(img)

        plot_images([img, combined, lane_map],
                    ["original : {}".format(filename), "combined", "lane_map"])


