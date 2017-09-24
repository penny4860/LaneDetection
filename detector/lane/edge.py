# -*- coding: utf-8 -*-
import cv2
import numpy as np

class _EdgeExtractor(object):
    
    def __init__(self, min_thd=50, max_thd=200):
        self._threshold = (min_thd, max_thd)
    
    def run(self, image):
        """
        # Args
            image : 3d array 
                RGB-ordered image
            min_thd : int
                minimum threshold
            max_thd : int
                maximum threshold
        
        # Returns
            edges : 2d array
                binary format edge map. 
                (edge pixel : 255, non-edge pixel : 0)
        """
        pass

class CannyEdgeExtractor(_EdgeExtractor):
    def run(self, image):
        edges = cv2.Canny(image,
                          self._threshold[0],
                          self._threshold[1])
        return edges


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
    import matplotlib.pyplot as plt
    import glob
    
    edge_extractor = CannyEdgeExtractor(50, 200)
    files = glob.glob('..//test_images//*.jpg')
    for filename in files[:1]:
        img = plt.imread(filename)
        edges = edge_extractor.run(img)
        plt.imshow(edges, cmap="gray")
        plt.show()
    
    # Todo : MidTwoEdges / EdgeDistanceCalculator 에 대한 client code





        
        
