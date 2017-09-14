# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


class DistortionCorrector(object):
    def __init__(self, images, n_pattern_rows=6, n_pattern_cols=9):
        """
        # Args
            images : list of 2d arrays
                gray scale images
            n_pattern_rows : int
                number of corner points in row(y) axis
            n_pattenr_cols : int
                number of corner points in column(x) axis
        """
        
        self._img_points = self._get_img_points(images, n_pattern_rows, n_pattern_cols)
        self._obj_points = self._get_obj_points(len(self._img_points), n_pattern_rows, n_pattern_cols)
    
    def _get_img_points(self, images, n_pattern_rows, n_pattern_cols):
        """
        # Returns
            points : ndarray, shape of (n_images, n_pattern_rows*n_pattern_cols, 3)
                2d-image coordinate points
        """

        imgpoints = [] # 2d points in image plane.
        
        # Step through the list and search for chessboard corners
        for img in images:
        
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(img, (n_pattern_cols,n_pattern_rows), None)
        
            # If found, add object points, image points
            if ret == True:
                imgpoints.append(corners)
                
        return imgpoints
    
    def _get_obj_points(self, n_images, n_pattern_rows, n_pattern_cols):
        """
        # Returns
            points : ndarray, shape of (n_images, n_pattern_rows*n_pattern_cols, 3)
                3d-world coordinate points
        """
        point = np.zeros((n_pattern_rows*n_pattern_cols,3), np.float32)
        point[:,:2] = np.mgrid[0:n_pattern_cols, 0:n_pattern_rows].T.reshape(-1,2)
    
        points = [point for _ in range(n_images)]
        return points
    
    def run(self, image):
        img_size = (image.shape[1], image.shape[0]) # (w, h)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self._obj_points, self._img_points, img_size, None, None)
        
        if ret:
            dst = cv2.undistort(img, mtx, dist, None, mtx)
        else:
            raise Exception('fail to distortion correction!')
        return dst


# Make a list of calibration images
files = glob.glob('camera_cal/*.jpg')
images = [cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2GRAY) for image_file in files]

corrector = DistortionCorrector(images, n_pattern_rows=6, n_pattern_cols=9)

# 3. Do calibration for test image size
img = cv2.imread('camera_cal/calibration1.jpg')
dst = corrector.run(img)

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()

