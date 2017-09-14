# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("test_images/straight_lines1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
img = img[:,:,2]

plt.imshow(img, cmap="gray")
plt.show()


