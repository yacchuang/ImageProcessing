# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:05:33 2023

@author: Ya-Chen.Chuang
"""

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/HandEcompressed_Scan1.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img)

img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2


from skimage.filters.rank import entropy
from skimage.morphology import disk

entropy_img = entropy(img, disk(1))
entropy1 = entropy_img.reshape(-1)
df['entropy'] = entropy1

plt.imshow(entropy_img)


from scipy import ndimage as nd

gaussian_image = nd.gaussian_filter(img, sigma=3)

plt.imshow(gaussian_image)

gaussian1 = gaussian_image.reshape(-1)

df['Gaussian'] = gaussian1


from skimage.filters import sobel

sobel_img = sobel(img)

plt.imshow(sobel_img)

sobel1 = sobel_img.reshape(-1)

df['Sobel'] = sobel1

print(df)