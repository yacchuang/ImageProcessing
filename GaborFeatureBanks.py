# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:48:08 2023

@author: Ya-Chen.Chuang
"""

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt


img = cv2.imread("C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/HandEcompressed_Scan1.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)

