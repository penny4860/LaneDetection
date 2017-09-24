# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
import cv2

# test5.jpg
import numpy as np


class SlidingWindow(object):
    
    def __init__(self):
        pass

    def run(self, lane_map):
        """
        # Args
            lane_map : array
                bird eye's view binary image
                
        # Returns
            out_img
            left_pixels
            right_pixels
        """
        self._lane_map = lane_map
        
        # 1. Create an output image to draw on and  visualize the result
        self._out_img = np.dstack((lane_map, lane_map, lane_map)).astype(np.uint8)
    
        # 2. Step through the windows one by one
        left_lane_inds, right_lane_inds, nonzerox, nonzeroy = self._run_sliding_window()
        
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        left_pixels = np.concatenate([leftx.reshape(-1,1), lefty.reshape(-1,1)], axis=1)
        right_pixels = np.concatenate([rightx.reshape(-1,1), righty.reshape(-1,1)], axis=1)
        return self._out_img, left_pixels, right_pixels

    def _get_start_window(self, nwindows):
        leftx_base, rightx_base = self._get_base(self._lane_map)
        # Set height of windows
        window_height = np.int(self._lane_map.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self._lane_map.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        return window_height, nonzerox, nonzeroy, leftx_current, rightx_current, left_lane_inds, right_lane_inds

    def _get_base(self, image):
        roi = image[image.shape[0]//2:,:]
        histogram = np.sum(roi, axis=0)

        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base

    def _run_sliding_window(self, nwindows=9, margin=150, minpix=10):
        
        window_height, nonzerox, nonzeroy, leftx_current, rightx_current, left_lane_inds, right_lane_inds = self._get_start_window(nwindows)
        lane_map = self._lane_map

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = lane_map.shape[0] - (window+1)*window_height
            win_y_high = lane_map.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(self._out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
            cv2.rectangle(self._out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        return left_lane_inds, right_lane_inds, nonzerox, nonzeroy


class LaneCurveFit(object):
    def __init__(self):
        pass

    def run(self, left_pixels, right_pixels):
        """
        # Args
            lane_map : array
                bird eye's view binary image
            nwindows : int
                number of windows
            margin : int
                the width of the windows +/- margin
            minpix : int
                minimum number of pixels found to recenter window
        """
        left_x = left_pixels[:, 0]
        left_y = left_pixels[:, 1]

        right_x = right_pixels[:, 0]
        right_y = right_pixels[:, 1]

        # Fit a second order polynomial to each
        self._left_fit = np.polyfit(left_y, left_x, 2)
        self._right_fit = np.polyfit(right_y, right_x, 2)
        

    def plot(self, out_img, left_pixels, right_pixels):
        # Generate x and y values for plotting
        ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
        left_fitx = self._left_fit[0]*ploty**2 + self._left_fit[1]*ploty + self._left_fit[2]
        right_fitx = self._right_fit[0]*ploty**2 + self._right_fit[1]*ploty + self._right_fit[2]
         
        out_img[left_pixels[:, 1], left_pixels[:, 0]] = [255, 0, 0]
        out_img[right_pixels[:, 1], right_pixels[:, 0]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()


class Curvature(object):

    def __init__(self, xm_per_pix=3.7/700, ym_per_pix=30/720):
        # meters per pixel in x, y dimension
        self._xm_per_pix = xm_per_pix 
        self._ym_per_pix = ym_per_pix 

    def calc(self, left_pixels, right_pixels):
        def _calc(y_eval, coef):
            curverad = ((1 + (2*coef[0]*y_eval + coef[1])**2)**1.5) / np.absolute(2*coef[0])
            return curverad

        left_meters = self._pixel_to_meters(left_pixels)
        right_meters = self._pixel_to_meters(right_pixels)

        left_fit = np.polyfit(left_meters[:, 1], left_meters[:, 0], 2)
        right_fit = np.polyfit(right_meters[:, 1], right_meters[:, 0], 2)

        # Calculate the new radii of curvature : Now our radius of curvature is in meters
        left_curverad = _calc(np.max(left_meters[:, 1]), left_fit)
        right_curverad = _calc(np.max(right_meters[:, 1]), right_fit)
        return left_curverad, right_curverad

    def _pixel_to_meters(self, pixels):
        xs_pixels = pixels[:, 0]
        ys_pixels = pixels[:, 1]
        
        xs_meters = xs_pixels * self._xm_per_pix
        ys_meters = ys_pixels * self._ym_per_pix
        
        meters = np.concatenate([xs_meters.reshape(-1,1), ys_meters.reshape(-1,1)], axis=1)
        return meters


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
        if plot:
            plt.imshow(result)
            plt.show()
        return result

    def _generate_pts(self, height, left_curve, right_curve):
        ys = np.linspace(0, height-1, height)
        left_xs = left_curve[0]*ys**2 + left_curve[1]*ys + left_curve[2]
        right_xs = right_curve[0]*ys**2 + right_curve[1]*ys + right_curve[2]
        return ys, left_xs, right_xs

if __name__ == "__main__":

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
    
    from detector.curve.warp import LaneWarper
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
    
    

