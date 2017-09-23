# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from detector.cal import DistortionCorrector
from detector.warp import PerspectTrans
import cv2
from detector.imutils import plot_images
from detector.lane import LanePixelDetector
from detector.binary import SchannelBin
from detector.imutils import closing

# test5.jpg
import numpy as np
np.set_printoptions(linewidth=1000, edgeitems=1000)

def get_roi(image):
    # bottom half of the image
    return image[image.shape[0]//2:,:]

def get_base(histogram):
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base

def add_square_feature(X):
    X = np.concatenate([(X**2).reshape(-1,1), X.reshape(-1,1)], axis=1)
    return X


def run_framework(image):
    corrector = DistortionCorrector.from_pkl("dataset//distortion_corrector.pkl")
    detector = LanePixelDetector()
    binarizer = SchannelBin()

    # 1. distortion correction    
    image = corrector.run(image)
    
    # 2. edge
    edges = cv2.Canny(image, 50, 200)
    
    # 3. binary
    binary_img = binarizer.run(img, (48, 255))
    binary_img = binarizer.roi_mask(binary_img)
    binary_img = closing(binary_img)

    # 4. lane map
    lane_map = detector.run(edges, binary_img)
    return lane_map

if __name__ == "__main__":

    # 2. Thresholding
    img = plt.imread('test_images/test1.jpg')
    lane_map = run_framework(img)
    translator = PerspectTrans.from_pkl('dataset/perspective_trans.pkl')
    
    binary_warped = translator.run(lane_map)
    plot_images([img, binary_warped, lane_map],
                ["original", "thresholded", "lane_map"])

    # 1. Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype(np.uint8)

    # 2. Get Histogram
    roi = get_roi(binary_warped)
    histogram = np.sum(roi, axis=0)
    leftx_base, rightx_base = get_base(histogram)
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 150
    # Set minimum number of pixels found to recenter window
    minpix = 10
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
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
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
#     from sklearn import linear_model
#     ransac = linear_model.RANSACRegressor()
#     ransac.fit(add_square_feature(righty), rightx)
#     
#     right_fit = ransac.estimator_.coef_.tolist()
#     right_fit.append(ransac.estimator_.intercept_)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    
    plt.show()
    

