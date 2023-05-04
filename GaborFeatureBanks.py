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

img2 = img.reshape(-1)

df = pd.DataFrame()
df['original image'] = img2

'''
GaborFilter = cv2.getGaborKernel((50, 50), 1, 1 / 4. * np.pi, np.pi, 0.95, 0, ktype=cv2.CV_32F)
fimg = cv2.filter2D(img2, cv2.CV_8UC3, GaborFilter)
filter_img = fimg.reshape(img.shape[0],img.shape[1])
plt.imshow(filter_img)
'''

# Generate Gabor features
num = 1

for sigma in (1,3):
    for theta in range(4):
        theta = theta/ 4. * np.pi
        for lamda in np.arange(0, np.pi, np.pi / 4): 
            for gamma in (0.05, 0.5, 0.95):
                Gabor_label = "Gabor" + str(num)
                
                GaborFilter = cv2.getGaborKernel((50, 50), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                
                
                fimg = cv2.filter2D(img2, cv2.CV_8UC3, GaborFilter)
                df[Gabor_label] = fimg
                print(Gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                
                num += 1
                


df.to_csv("GaborPractice.csv")
