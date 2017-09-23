# -*- coding: utf-8 -*-

import numpy as np
import cv2
import pickle


class PerspectTrans(object):
    
    def __init__(self, dst_size):
        """
        # Args
            dst_size : tuple
                (w, h)
        """
        self._src = np.array([(250, 700), (1075, 700), (600, 450), (685, 450)]).astype(np.float32)

        w, h = dst_size

        x_offset = 300
        y_offset = 50
        
        dst = np.array([(x_offset, h-y_offset),
                        (w-x_offset, h-y_offset),
                        (x_offset, y_offset),
                        (w-x_offset, y_offset)]).astype(np.float32)
        
        self._M = cv2.getPerspectiveTransform(self._src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, self._src)
    
    def run(self, image):
        h, w = image.shape[:2]
        warped = cv2.warpPerspective(image, self._M, (w, h), flags=cv2.INTER_LINEAR)
        return warped
    
    def to_pkl(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def from_pkl(cls, filename):
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        return instance
