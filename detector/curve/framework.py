# -*- coding: utf-8 -*-

from detector.curve.pers import LaneWarper, LaneMarker
from detector.curve.curv import Curvature
from detector.curve.fit import SlidingWindow, LaneCurveFit

class LaneFitFramework(object):
    
    def __init__(self, warper=LaneWarper(), window=SlidingWindow(), fitter=LaneCurveFit()):
        self._warper = warper
        self._window = window
        self._fitter = fitter
    
    def run(self, lane_map):
        warper = LaneWarper()
        lane_map_ipt = warper.forward(lane_map)
    
        win = SlidingWindow()
        out_img, left_pixels, right_pixels = win.run(lane_map_ipt)
        fitter = LaneCurveFit()
        fitter.run(left_pixels, right_pixels)
        fitter.plot(out_img, left_pixels, right_pixels)
         
        curv = Curvature()
        l, r = curv.calc(left_pixels, right_pixels)
        print(l, 'm', r, 'm')
        
        marker = LaneMarker(warper)
        marker.run(undist_img, fitter._left_fit, fitter._right_fit)
    
    def _show_process(self):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from detector.cal import DistortionCorrector
    
    # 1. Get bird eye's view lane map
    img = plt.imread('../../test_images/straight_lines1.jpg')
    img = plt.imread('../../test_images/test6.jpg')

    corrector = DistortionCorrector.from_pkl("..//..//dataset//distortion_corrector.pkl")

    # lane_map_ipt = run_framework(img)
    from detector.lane.lane import LaneDetector
    from detector.lane.edge import CannyEdgeExtractor
    from detector.lane.mask import LaneImageMask
    from detector.lane.binary import SchannelBin
    _edge_detector = CannyEdgeExtractor(50, 200)
    _binary_extractor = SchannelBin((48, 255))
    _image_mask = LaneImageMask()
    detector = LaneDetector(_edge_detector, _binary_extractor, _image_mask)

    undist_img = corrector.run(img)
    lane_map = detector.run(undist_img)
    
    #####################################################################################
    frm = LaneFitFramework()
    frm.run(lane_map)
    
#     warper = LaneWarper()
#     lane_map_ipt = warper.forward(lane_map)
# 
#     win = SlidingWindow()
#     out_img, left_pixels, right_pixels = win.run(lane_map_ipt)
#     fitter = LaneCurveFit()
#     fitter.run(left_pixels, right_pixels)
#     fitter.plot(out_img, left_pixels, right_pixels)
#      
#     curv = Curvature()
#     l, r = curv.calc(left_pixels, right_pixels)
#     print(l, 'm', r, 'm')
#     
#     marker = LaneMarker(warper)
#     marker.run(undist_img, fitter._left_fit, fitter._right_fit)
