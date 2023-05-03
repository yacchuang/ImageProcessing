# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:03:40 2023

@author: Ya-Chen.Chuang
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/HandEcompressed_Scan1.tif')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)
plt.imshow(img)


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))


plt.imshow(res2)


cv.imshow('res2', res2)
cv.waitKey(0)
cv.destroyAllWindows()