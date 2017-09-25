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


class LaneMarker(object):
    def __init__(self, warper):
        self._warper = warper
    
    def run(self, image, left_fit, right_fit, plot=False):
        """
        # Args
            image : distortion corrected image
        """
        ploty, left_fitx, right_fitx = self._generate_pts(image.shape[0], left_fit, right_fit)

        color_warp = np.zeros_like(image).astype(np.uint8)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self._warper.backward(color_warp)
    
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        # Todo : return
        self.right_pixels = np.array(right_fitx, ploty).astype(int).T
        self.left_pixels = np.array(left_fitx, ploty).astype(int).T
        
        if plot:
            plt.imshow(result)
            plt.show()
        return result

    def _generate_pts(self, height, left_curve, right_curve):
        ys = np.linspace(0, height-1, height)
        left_xs = left_curve[0]*ys**2 + left_curve[1]*ys + left_curve[2]
        right_xs = right_curve[0]*ys**2 + right_curve[1]*ys + right_curve[2]
        return ys, left_xs, right_xs

    def get_lane_line_map(self, image):
        line_map = np.zeros((image.shape[0], image.shape[1])).astype(np.uint8)
        line_map[self.left_pixels[:, 1], self.left_pixels[:, 0]] = 255
        line_map[self.right_pixels[:, 1], self.right_pixels[:, 0]] = 255
        line_map = self._warper.backward(line_map)
        return line_map

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

