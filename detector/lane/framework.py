# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import cv2
import numpy as np

from detector.cal import DistortionCorrector
from detector.lane.binary import SchannelBin
from detector.lane.edge import CannyEdgeExtractor, EdgeDistanceCalculator, MidTwoEdges
from detector.lane.mask import LaneImageMask


class LaneDetectionFramework(object):
    """Detect lane pixels using edge map & binay map
    
    # Args
        edge_detector : _EdgeExtractor
        binary_extractor : _BinExtractor
        image_mask : ImageMask
    """
    
    _VALID_PIXEL = 255
    
    def __init__(self,
                 edge_detector=CannyEdgeExtractor(50, 200),
                 binary_extractor=SchannelBin((48, 255)),
                 image_mask=LaneImageMask()):
        # Instance injected from outside
        self._edge_detector = edge_detector
        self._bin_extractor = binary_extractor
        self._img_mask = image_mask
        
        # Instance created internally
        self._edge_dist_calc = EdgeDistanceCalculator()
        self._mid_edge_calc = MidTwoEdges()
     
    def run(self, image, plot=False):
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
        
        if plot:
            self._show_process(image, edge_map, binary_roi, lane_map)
        
        return lane_map

    def _show_process(self, img, edge_map, binary, lane):
        def _plot_images(images, titles):
            _, axes = plt.subplots(1, len(images), figsize=(10,10))
            for img, ax, text in zip(images, axes, titles):
                ax.imshow(img, cmap="gray")
                ax.set_title(text, fontsize=30)
            plt.show()

        combined = np.zeros_like(img)
        combined[:,:,0] += edge_map
        combined[:,:,2] += binary

        # original / binary / edge / combined / lane
        _plot_images([img, combined, lane], ["input", "binary(Blue) & edge(Red)", "lane"])
        

if __name__ == "__main__":
    corrector = DistortionCorrector.from_pkl("..//..//dataset//distortion_corrector.pkl")
    detector = LaneDetectionFramework()

    # 1. Distortion Correction
    import glob
    files = glob.glob('..//..//test_images//*.jpg')
    for filename in files[:-1]:
        img = plt.imread(filename)
        img = corrector.run(img)
        
        lane_map = detector.run(img, True)

