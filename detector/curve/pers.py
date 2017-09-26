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
    
    def forward(self, image, plot=False):
        """src to dst"""
        h, w = image.shape[:2]
        warped = cv2.warpPerspective(image, self._M, (w, h), flags=cv2.INTER_LINEAR)
        
        if plot:
            self._show_process(image, warped)
        return warped

    def backward(self, image):
        """dst to src"""
        h, w = image.shape[:2]
        warped = cv2.warpPerspective(image, self._Minv, (w, h), flags=cv2.INTER_LINEAR)
        return warped

    def _show_process(self, original_image, transformed_image):
        
        img = original_image.copy()
        for point in self._src:
            cv2.circle(img, center=(point[0], point[1]), radius=5, thickness=20, color=(0,0,255))
        
        img_tr = transformed_image.copy()
        for point in self._dst:
            cv2.circle(img_tr, center=(point[0], point[1]), radius=5, thickness=20, color=(0,0,255))
        
        _, axes = plt.subplots(1, 2, figsize=(10,10))
        for img, ax, text in zip([img, img_tr], axes, ["img", "bird eyes view"]):
            ax.imshow(img, cmap="gray")
            ax.set_title(text, fontsize=30)
        plt.show()

        

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
    def __init__(self, warper, curvature):
        self._warper = warper
        self._curvature = curvature
    
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

        offset = self._get_vehicle_offset(newwarp)
        self._create_image(result, offset)
        
        if plot:
            plt.imshow(result)
            plt.show()
        return result

    def _generate_pts(self, height, left_curve, right_curve):
        ys = np.linspace(0, height-1, height)
        left_xs = left_curve[0]*ys**2 + left_curve[1]*ys + left_curve[2]
        right_xs = right_curve[0]*ys**2 + right_curve[1]*ys + right_curve[2]
        return ys, left_xs, right_xs

    def _get_vehicle_offset(self, new_warp):
        array = new_warp[-1,:,1]
        bottom_x_left = np.where(array != 0)[0][0]
        bottom_x_right = np.where(array != 0)[0][-1]
        
        vehicle_offset = (bottom_x_left + bottom_x_right)/2 - new_warp.shape[1]/2
        # Convert pixel offset to meters
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        vehicle_offset *= xm_per_pix
        
        return vehicle_offset

    def _create_image(self, result, vehicle_offset):
        left_curv, right_curv = self._curvature
         
        # Annotate lane curvature values and vehicle offset from center
        avg_curve = (left_curv + right_curv)/2
        curv_str = 'Radius of curvature: {:.1f}m'.format(avg_curve)
        result = cv2.putText(result, curv_str, (30,40), 0, 1, (255,255,255), 2, cv2.LINE_AA)
        
        if vehicle_offset > 0:
            offset_str = 'Vehicle is {:.2f}m right of lane center'.format(vehicle_offset)
        else:
            offset_str = 'Vehicle is {:.2f}m left of lane center'.format(-1*vehicle_offset)
        result = cv2.putText(result, offset_str, (30,70), 0, 1, (255,255,255), 2, cv2.LINE_AA)
        return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    warper = LaneWarper()
    img = plt.imread('..//..//test_images/straight_lines1.jpg')
    img_bird = warper.forward(img, True)
    
    

#     _, axes = plt.subplots(1, 3, figsize=(10,10))
#     for img, ax, text in zip([img, img_bird], axes, ["img", "bird eyes view"]):
#         ax.imshow(img, cmap="gray")
#         ax.set_title(text, fontsize=30)
#     plt.show()

