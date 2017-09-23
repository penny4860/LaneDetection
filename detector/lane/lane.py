# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
import cv2
import numpy as np


np.set_printoptions(linewidth=500000)


class LaneDetector(object):
    """Detect lane pixels using edge map & binay map
    
    # Args
        edge_detector : _EdgeExtractor
        binary_extractor : _BinExtractor
        image_mask : ImageMask
    """
    
    _VALID_PIXEL = 255
    
    def __init__(self, edge_detector, binary_extractor, image_mask):
        # Instance injected from outside
        self._edge_detector = edge_detector
        self._bin_extractor = binary_extractor
        self._img_mask = image_mask
        
        # Instance created internally
        self._edge_dist_calc = EdgeDistanceCalculator()
        self._mid_edge_calc = MidTwoEdges()
     
    def run(self, image):
        """
        # Args
            image : 3d array
                RGB ordered image
            
        # Returns
            lane_map : 2d array
                lane pixel detector binary image
        """
        
        edge_map = self._edge_detector.run(image)
        binary = self._bin_extractor.run(image)
        binary_roi = self._img_mask.run(binary)
        # binary_img = closing(binary_img)
        
        # 1. For the binary image, get the right & left edge distance map
        r_dist_map, l_dist_map = self._edge_dist_calc.run(edge_map, binary_roi)

        # 2. Get middle pixels of two edges
        lane_map = self._mid_edge_calc.run(r_dist_map, l_dist_map)
        return lane_map


class EdgeDistanceCalculator(object):
    def __init__(self):
        pass
    
    def run(self, edge_map, binary_map):
        """For the bright pixels get the nearest edge distance.

        # Args
            edge_map : 2d array
            binary_map : 2d array
        
        # Returns
            right_dist_map : 2d array
                nearest right edge distance map

            left_dist_map : 2d array
                nearest right edge distance map
        """
        bright_pixels = np.where(binary_map != 0)

        left_dist_map = np.zeros_like(edge_map).astype(float) - 1
        right_dist_map = np.zeros_like(edge_map).astype(float) - 1

        for r, c in zip(bright_pixels[0], bright_pixels[1]):
            r_dist = self._calc_right_edge_dist(r, c, edge_map)
            right_dist_map[r, c] = r_dist

            l_dist = self._calc_left_edge_dist(r, c, edge_map)
            left_dist_map[r, c] = l_dist

        return right_dist_map, left_dist_map

    def _calc_right_edge_dist(self, y, x, edge_map):
        array = edge_map[y, x:]
        dist = self._smallest_non_zero_index(array)
        return dist

    def _calc_left_edge_dist(self, y, x, edge_map):
        array = edge_map[y, :x+1][::-1]
        dist = self._smallest_non_zero_index(array)
        return dist

    def _smallest_non_zero_index(self, array):
        """Get smallest non-zero index from an 1d array
        
        # Args
            array : 1d array
        # Returns
            dist : smallest non-zero index
        """
        indices = np.where(array != 0)[0]
        if indices.size == 0:
            dist = np.inf
        else:
            dist = indices[0]
        return dist

# Center of both edges
class MidTwoEdges(object):
    
    _VALID_PIXEL = 255
    
    def __init__(self, width_thd=30 , mid_point_thd=3):
        self._width_thd = width_thd
        self._mid_point_thd = mid_point_thd
    
    def run(self, right_dist_map, left_dist_map):
        mid_of_edges = self._get_mid_of_edges(right_dist_map, left_dist_map)
        extended_mids = self._extend_mid_to_edges(mid_of_edges, right_dist_map, left_dist_map)
        return extended_mids
        
    def _get_mid_of_edges(self, right_dist_map, left_dist_map):
        """Get middle point pixels of right & left edges
        
        # Args
            right_dist_map : 2d array
            left_dist_map : 2d array
        
        # Returns
            lane_map : 2d array
        """
        # Todo : image의 row 위치에 따라서 r_dist_map+l_dist_map threshold 를 linear 하게 적용하자.
        mid_of_edges = np.zeros_like(right_dist_map)

        # Get mid-point from right & left edge pixels, 
        # and if its width is smaller than threshold,
        mid_of_edges[(abs(right_dist_map - left_dist_map) <= self._mid_point_thd) & 
                 (right_dist_map > 0) & 
                 (left_dist_map > 0) & 
                 (right_dist_map+left_dist_map < self._width_thd)] = self._VALID_PIXEL
        return mid_of_edges

    def _extend_mid_to_edges(self, lane_map, right_dist_map, left_dist_map):
        """Extend mid pixels to its nearset edge"""
        
        lane_pixels = np.where(lane_map == self._VALID_PIXEL)
        for r, c in zip(lane_pixels[0], lane_pixels[1]):
            right_dist = int(right_dist_map[r, c])
            left_dist = int(left_dist_map[r, c])
            lane_map[r, c:c+right_dist] = self._VALID_PIXEL
            lane_map[r, c-left_dist+1:c] = self._VALID_PIXEL
        return lane_map


if __name__ == "__main__":
    from detector.imutils import plot_images
    from detector.lane.binary import SchannelBin
    from detector.lane.edge import CannyEdgeExtractor
    from detector.lane.mask import LaneImageMask
    
    corrector = DistortionCorrector.from_pkl("..//..//dataset//distortion_corrector.pkl")

#     combined = np.zeros_like(img)
#     combined[:,:,0] += edges
#     combined[:,:,2] += binary_img
#     combined = region_of_interest(combined)

    _edge_detector = CannyEdgeExtractor(50, 200)
    _binary_extractor = SchannelBin((48, 255))
    _image_mask = LaneImageMask()

    detector = LaneDetector(_edge_detector, _binary_extractor, _image_mask)

    # 1. Distortion Correction
    import glob
    files = glob.glob('..//..//test_images//*.jpg')
    for filename in files[:1]:
        img = plt.imread(filename)
        img = corrector.run(img)
        
        lane_map = detector.run(img)
        plot_images([img, lane_map],
                    ["original : {}".format(filename), "lane_map"])


