# -*- coding: utf-8 -*-
import cv2


class _EdgeExtractor(object):
    
    def __init__(self, min_thd=50, max_thd=200):
        self._threshold = (min_thd, max_thd)
    
    def run(self, image):
        """
        # Args
            image : 3d array 
                RGB-ordered image
            min_thd : int
                minimum threshold
            max_thd : int
                maximum threshold
        
        # Returns
            edges : 2d array
                binary format edge map. 
                (edge pixel : 255, non-edge pixel : 0)
        """
        pass

class CannyEdgeExtractor(_EdgeExtractor):
    def run(self, image):
        edges = cv2.Canny(image,
                          self._threshold[0],
                          self._threshold[1])
        return edges


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import glob
    
    edge_extractor = CannyEdgeExtractor(50, 200)
    files = glob.glob('..//test_images//*.jpg')
    for filename in files[:1]:
        img = plt.imread(filename)
        edges = edge_extractor.run(img)
        plt.imshow(edges, cmap="gray")
        plt.show()






        
        
