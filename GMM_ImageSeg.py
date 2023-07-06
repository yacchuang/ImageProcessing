# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:07:04 2023

@author: Ya-Chen.Chuang
"""

import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tifffile import imsave

img = cv2.imread('R:/YaChen/TrainingImage/image2/C4-Open3-2ndExp_Scan1.qptiff-1.tif')
Z = img.reshape((-1,3))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = img.reshape(-1)

from sklearn.mixture import GaussianMixture as GMM

# Pick the optimal number using BIC
n_components = np.arange(1,5)
GMMmodel = [GMM(n, covariance_type='tied').fit(Z) for n in n_components]

plt.plot(n_components, [m.bic(Z) for m in GMMmodel], label = "BIC")
plt.xlabel("n_components")


# Pick n_components based on the curve, somewhere around elbow
GMMmodel = GMM(n_components=3, covariance_type='tied').fit(Z)
GMMlabel = GMMmodel.predict(Z)


# Extract label for mask
df_mask = GMMlabel
df_img = img2
NewImg = []
df = pd.DataFrame()

row = 0
for index in df_mask: 
    if index == 1 or index == 2:
        NewImg.append("255")
    else:
        NewImg.append("0")
    
    row += 1

df['NewImage'] = NewImg

ImageArray = np.array(df['NewImage'].array)
Image = ImageArray.astype(np.uint8)
NewImg = Image.reshape(img.shape)
plt.imshow(NewImg)
imsave("C4-Open3-2ndExp_Scan1.qptiff-1_GMM_all_seg_mask.tif", NewImg)

'''
original_shape = img.shape
SegImage = GMMlabel.reshape(original_shape[0], original_shape[1])
plt.imshow(SegImage)
imsave("C4-Open3-2ndExp_Scan1.qptiff-2_GMM_all_seg.tif", SegImage)
'''