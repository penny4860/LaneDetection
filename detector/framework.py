# -*- coding: utf-8 -*-

from detector.cal import DistortionCorrector
from detector.lane.framework import LaneDetectionFramework
from detector.curve.framework import LaneFitFramework

class ImageFramework(object):
    """
    # Args
        dist_corrector
        lane_detector
        warper
        lane_fitter
        lane_painter
    """
    
    def __init__(self,
                 corrector=DistortionCorrector.from_pkl(),
                 detector=LaneDetectionFramework(),
                 fitter = LaneFitFramework()):
        self._corrector = corrector
        self._detector = detector
        self._fitter = fitter
    
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
        undist_img = self._corrector.run(image)
        lane_map = self._detector.run(undist_img)
        lane_marked_img = self._fitter.run(undist_img, lane_map)
        return lane_marked_img

    def _show_process(self):
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
    
    
