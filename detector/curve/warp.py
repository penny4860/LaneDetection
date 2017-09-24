# -*- coding: utf-8 -*-

import numpy as np
import cv2


class Warper(object):
    """Perform perspective transform"""
    
    def __init__(self, src_points, dst_points):
        """
        # Args
            dst_size : tuple
                (w, h)
        """
        self._src = src_points.astype(np.float32)
        self._dst = dst_points.astype(np.float32)
        
        self._M = cv2.getPerspectiveTransform(self._src, self._dst)
        self._Minv = cv2.getPerspectiveTransform(self._dst, self._src)
    
    def forward(self, image):
        """src to dst"""
        h, w = image.shape[:2]
        warped = cv2.warpPerspective(image, self._M, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    def backward(self, image):
        """dst to src"""
        h, w = image.shape[:2]
        warped = cv2.warpPerspective(image, self._Minv, (w, h), flags=cv2.INTER_LINEAR)
        return warped

class LaneWarper(Warper):
    """Perform perspective transform to make a image to bird eye's view"""

    def __init__(self, src_points=None, dst_points=None, dst_size=(1280, 720)):
        
        if src_points is None:
            src_points = np.array([(250, 700), (1075, 700), (600, 450), (685, 450)]).astype(np.float32)
        if dst_points is None:
            w, h = dst_size
            x_offset = 300
            y_offset = 50
            dst_points = np.array([(x_offset, h-y_offset),
                                   (w-x_offset, h-y_offset),
                                   (x_offset, y_offset),
                                   (w-x_offset, y_offset)]).astype(np.float32)
        
        super(LaneWarper, self).__init__(src_points, dst_points)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    warper = LaneWarper()
    img = plt.imread('..//..//test_images/straight_lines1.jpg')
    img_bird = warper.forward(img)
    original = warper.backward(img_bird)

    _, axes = plt.subplots(1, 3, figsize=(10,10))
    for img, ax, text in zip([img, img_bird, original], axes, ["img", "bird eyes view", "original"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(text, fontsize=30)
    plt.show()

