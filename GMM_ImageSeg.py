# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:07:04 2023

@author: Ya-Chen.Chuang
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/HandEcompressed_Scan1.tif')
Z = img.reshape((-1,3))

from sklearn.mixture import GaussianMixture as GMM

GMMmodel = GMM(n_components=3, covariance_type='tied').fit(Z)
GMMlabel = GMMmodel.predict(Z)

original_shape = img.shape
SegImage = GMMlabel.reshape(original_shape[0], original_shape[1])
plt.imshow(SegImage)

