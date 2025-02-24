# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:03:40 2023

@author: Ya-Chen.Chuang
"""

import pandas as pd
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tifffile import imsave


img = cv.imread('R:/YaChen/TrainingImage/DotImg/C2-SP-013908_6001 10_SR-ASO-DMD5-S1_Mfa-PECAM1-C2_Mfa-S100b-C3_Mfa-GAP43-C4_wygr_01.tif')
# Z = img.reshape((-1,3))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img2 = img.reshape(-1)

# convert to np.float32
img2 = np.float32(img2)
plt.imshow(img)


# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv.kmeans(img2,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

plt.imshow(res2)

'''
cv.imshow('res2', res2)
cv.waitKey(0)
cv.destroyAllWindows()
'''

# Extract label for mask
df_mask = label
df_img = img2
NewImg = []
df = pd.DataFrame()

row = 0
for index in df_mask: 
    if index == 2 or index == 1:
        NewImg.append("0")
    else:
        NewImg.append("255")
    
    row += 1

df['NewImage'] = NewImg

ImageArray = np.array(df['NewImage'].array)
Image = ImageArray.astype(np.uint8)
NewImg = Image.reshape(img.shape)
plt.imshow(NewImg)
imsave("C2-SP-013908_6001 10_SR-ASO-DMD5-S1_Mfa-PECAM1-C2_Mfa-S100b-C3_Mfa-GAP43-C4_wygr_Kmean_seg_mask_01.tif", NewImg)
