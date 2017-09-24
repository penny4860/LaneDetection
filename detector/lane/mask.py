# -*- coding: utf-8 -*-

import numpy as np
import cv2


class ImageMask(object):
    
    def __init__(self):
        pass
    
    def run(self, image, regions):
        """
        # Args
            image : 2d or 3d array
            regions : list of 2d array, shape of (n_points, 2)
                interested region points

        # Returns
            masked_image
        """
        for points in regions:
            image = self._roi_masking(image, points)
        return image

    def _roi_masking(self, image, points):
        #defining a blank mask to start with
        mask = np.zeros_like(image)
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, points, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image



class LaneImageMask(ImageMask):
    
    def run(self, image, regions=[]):
        ylength, xlength = image.shape[:2]
        default_region = self._set_default_vertices(ylength, xlength)
        extended_regions = regions + [default_region]
        return super(LaneImageMask, self).run(image, extended_regions)

    def _set_default_vertices(self, ylength, xlength):
        vertices = np.array([[(0, ylength),
                              (xlength/2-ylength/10, ylength*0.5),
                              (xlength/2+ylength/10, ylength*0.5),
                              (xlength, ylength)]], dtype=np.int32)
        return vertices


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import glob
    
    img_mask = LaneImageMask()
    files = glob.glob('..//test_images//*.jpg')
    for filename in files[:1]:
        img = plt.imread(filename)
        masked_img = img_mask.run(img)
        plt.imshow(masked_img, cmap="gray")
        plt.show()


