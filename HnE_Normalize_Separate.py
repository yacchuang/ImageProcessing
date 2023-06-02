# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:35:28 2023

@author: Ya-Chen.Chuang
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt


image = cv2.imread('C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/HandEcompressed_Scan1.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

Io = 240 # Transmitted light intensity, Normalizing factor for image intensities
alpha = 1  #As recommend in the paper. tolerance for the pseudo-min and pseudo-max (default: 1)
beta = 0.15 #As recommended in the paper. OD threshold for transparent pixels (default: 0.15)

######## Step 1: Convert RGB to OD ###################
## reference H&E OD matrix.
#Can be updated if you know the best values for your image. 
#Otherwise use the following default values. 
#Read the above referenced papers on this topic. 
HERef = np.array([[0.5626, 0.2159],
                  [0.7201, 0.8012],
                  [0.4062, 0.5581]])
### reference maximum stain concentrations for H&E
maxCRef = np.array([1.9705, 1.0308])