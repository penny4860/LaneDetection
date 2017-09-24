# -*- coding: utf-8 -*-

import numpy as np
import cv2


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


if __name__ == "__main__":
    pass
