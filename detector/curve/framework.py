# -*- coding: utf-8 -*-

from detector.curve.pers import LaneWarper, LaneMarker
from detector.curve.curv import Curvature
from detector.curve.fit import SlidingWindow, LaneCurveFit

class LaneFitFramework(object):
    
    def __init__(self, warper=LaneWarper(), window=SlidingWindow(), fitter=LaneCurveFit(), curv = Curvature()):
        self._warper = warper
        self._window = window
        self._fitter = fitter
        self._curv = curv
        
        # left / right curvature
        self.curvature = (None, None)
        
    def run(self, undist_img, lane_map, plot=False):
        
        # 1. Do perspective transform to make bird eyes view image
        lane_map_ipt = self._warper.forward(lane_map)

        # 2. Get lane pixels to fit lane curve    
        out_img, left_pixels, right_pixels = self._window.run(lane_map_ipt)

        # 3. Fit lane curve
        self._fitter.run(left_pixels, right_pixels)

        # 4. Calc curvature in meter unit         
        self.curvature = self._curv.calc(left_pixels, right_pixels)

        # 5. Mark lane area in original image        
        marker = LaneMarker(self._warper)
        marked_image = marker.run(undist_img, self._fitter._left_fit, self._fitter._right_fit)
        
        if plot:
            self._show_process(out_img, marked_image, left_pixels, right_pixels)
        return marked_image
    
    def _show_process(self, out_img, marked_image, left_pixels, right_pixels):
        # Todo : code cleaning
        self._fitter.plot(out_img, left_pixels, right_pixels)
        
        plt.imshow(marked_image)
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from detector.cal import DistortionCorrector
    
    # 1. Get bird eye's view lane map
    img = plt.imread('../../test_images/straight_lines1.jpg')
    img = plt.imread('../../test_images/test6.jpg')

    corrector = DistortionCorrector.from_pkl("..//..//dataset//distortion_corrector.pkl")

    # lane_map_ipt = run_framework(img)
    from detector.lane.framework import LaneDetectionFramework
    frm1 = LaneDetectionFramework()

    # 1. correction
    undist_img = corrector.run(img)
    lane_map = frm1.run(undist_img)
    #####################################################################################
    frm2 = LaneFitFramework()
    frm2.run(undist_img, lane_map, True)
    
