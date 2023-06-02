# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:55:03 2023

@author: Ya-Chen.Chuang
"""

import cv2
from skimage import io, data, img_as_ubyte
from skimage.filters import threshold_multiotsu
from matplotlib import pyplot as plt
import numpy as np


## Histogram Equalization and CLAHE
img = cv2.imread("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg-1/images/SP-014037_531426_Hs-APP-E9E10_1.tif")
LabImg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(LabImg)

plt.hist(l.flat, bins=100)

equ = cv2.equalizeHist(l)
plt.hist(equ.flat, bins=100)

equ_img = cv2.merge((equ,a,b))
hist_equ_img = cv2.cvtColor(equ_img, cv2.COLOR_LAB2BGR)


#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8,8))
clahe_img = clahe.apply(l)

CLAHE_img = cv2.merge((clahe_img, a, b))
hist_CLAHE_img = cv2.cvtColor(CLAHE_img, cv2.COLOR_LAB2BGR)

plt.imsave("SP-014037_531426_Hs-APP-E9E10_1.jpg", img)
plt.imsave("SP-014037_531426_Hs-APP-E9E10_1_Hist_Equal.jpg", hist_equ_img)
plt.imsave("SP-014037_531426_Hs-APP-E9E10_1_Hist_CLAHE.jpg", hist_CLAHE_img)


'''
## Multiotsu Segmentation
img = io.imread("C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/SP-014037_531426_Hs-APP-E9E10_1_Hist_CLAHE.jpg")

thresholds = threshold_multiotsu(img, classes=4)

regions = np.digitize(img, bins = thresholds)
output = img_as_ubyte(regions)
plt.imshow(output)
'''