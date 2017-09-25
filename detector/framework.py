# -*- coding: utf-8 -*-

import cv2
from detector.cal import DistortionCorrector

from detector.lane.edge import CannyEdgeExtractor, EdgeDistanceCalculator, MidTwoEdges
from detector.lane.binary import SchannelBin
from detector.lane.mask import LaneImageMask

from detector.curve.pers import LaneWarper, LaneMarker
from detector.curve.curv import Curvature
from detector.curve.fit import LaneCurveFit, SlidingWindow


class _LaneFramework(object):
    """
    # Args
        corrector
        edge_detector
        binary_extractor
        image_mask
        warper
        window
        fitter
        curv
    """
    
    def __init__(self,
                 corrector=DistortionCorrector.from_pkl(),
                 edge_detector=CannyEdgeExtractor(50, 200),
                 binary_extractor=SchannelBin((48, 255)),
                 image_mask=LaneImageMask(),
                 warper=LaneWarper(),
                 window=SlidingWindow(),
                 fitter=LaneCurveFit(),
                 curv=Curvature()):

        self._corrector = corrector
        self._edge_detector = edge_detector
        self._binary_extractor = binary_extractor
        self._image_mask = image_mask
        self._warper = warper
        self._window = window
        self._fitter = fitter
        self._curv = curv
        
        # Instance created internally
        self._edge_dist_calc = EdgeDistanceCalculator()
        self._mid_edge_calc = MidTwoEdges()
    
    def run(self, image):
        """
        # Args
            image : 3d array
                RGB ordered raw input image

        # Returns
            output_image : 3d array
                image marked lane area

            curvatures : tuple
                (left lane's curvature, right lane's curvature)
        """
        # 1. distortion correction
        undist_img = self._corrector.run(image)
        
        # 2. extract lane map
        lane_map = self._extract_lane_map(undist_img)
        lane_marked_img, _ = self._fit_lane_curve(undist_img, lane_map)
        
        return lane_marked_img

    def _extract_lane_map(self, image):
        # 2. edge detection
        edge_map = self._edge_detector.run(image)

        # 3. Get binary image
        binary = self._binary_extractor.run(image)
        binary_roi = self._image_mask.run(binary)
        
        # 4. Get lane map
        r_dist_map, l_dist_map = self._edge_dist_calc.run(edge_map, binary_roi)
        lane_map = self._mid_edge_calc.run(r_dist_map, l_dist_map)
        return lane_map

    def _fit_lane_curve(self, image, lane_map):
        # 5. Do perspective transform to make bird eyes view image
        lane_map_ipt = self._warper.forward(lane_map)

        # 6. Get lane pixels to fit lane curve    
        _, left_pixels, right_pixels = self._window.run(lane_map_ipt)

        # 7. Fit lane curve
        self._fitter.run(left_pixels, right_pixels)

        # 8. Calc curvature in meter unit         
        self.curvature = self._curv.calc(left_pixels, right_pixels)

        # 9. Mark lane area in original image
        marker = LaneMarker(self._warper)
        lane_marked_img = marker.run(image, self._fitter._left_fit, self._fitter._right_fit)
        line_img = marker.get_lane_line_map(image)
        return lane_marked_img, line_img
        
class VideoFramework(_LaneFramework):
     
    def __init__(self):
        self.roi = None
        super(VideoFramework, self).__init__()
 
    def run(self, image):
        # 1. distortion correction
        undist_img = self._corrector.run(image)
        
        # 2. extract lane map
        lane_map = self._extract_lane_map(undist_img)
        if self.roi is not None:
            lane_map[self.roi == 0] = 0
        
        lane_marked_img, lane_line_map = self._fit_lane_curve(undist_img, lane_map)
        self._set_roi(lane_line_map)
        
        return lane_marked_img

    def _set_roi(self, lane_line_map):
        import numpy as np
        kernel = np.ones((5,50), np.uint8)
        self.roi = cv2.dilate(lane_line_map,kernel,iterations = 1)


class ImageFramework(_LaneFramework):
    pass



if __name__ == "__main__":
    frm = ImageFramework()

    # 1. Distortion Correction
    import glob
    import matplotlib.pyplot as plt
    files = glob.glob('..//test_images//*.jpg')
    for filename in files[:-1]:
        img = plt.imread(filename)
        marked = frm.run(img)
        
        plt.imshow(marked)
        plt.show()
    
    
