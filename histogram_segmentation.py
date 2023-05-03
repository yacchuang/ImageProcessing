# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:31:20 2023

@author: Ya-Chen.Chuang
"""

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import io, img_as_ubyte, img_as_float
from skimage.color import rgb2gray
import numpy as np
from matplotlib import pyplot as plt

img = img_as_float(io.imread("C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/HandEcompressed_Scan1.tif"))

plt.imshow(img, cmap="gray")
gray = rgb2gray(img)


#Denoising and create a histogram
sigma_est = np.mean(estimate_sigma(gray, multichannel=True))
denoise = denoise_nl_means(gray, h=1.15*sigma_est, fast_mode=False, patch_size = 5, patch_distance=3, multichannel= False)

plt.imshow(denoise, cmap="gray")

denoise_ubyte = img_as_ubyte(denoise)
plt.imshow(denoise_ubyte, cmap="gray")

denoise_img_as_8byte = img_as_ubyte(denoise)
plt.hist(denoise_img_as_8byte.flat, bins=100, range=(0,255))


#Segmentation
seg1 = (denoise_ubyte <= 100)
seg2 = (denoise_ubyte > 100) & (denoise_ubyte <= 200)
seg3 = (denoise_ubyte > 200) & (denoise_ubyte <= 255)

all_seg = np.zeros((denoise_ubyte.shape[0], denoise_ubyte.shape[1], 3))

all_seg[seg1] = (1,0,0)
all_seg[seg2] = (0,1,0)
all_seg[seg3] = (0,0,1)
plt.imshow(all_seg)

#Cleaned Segmentation
from scipy import ndimage as nd

seg1_opened = nd.binary_opening(seg1, np.ones((3,3)))
seg1_closed = nd.binary_closing(seg1_opened, np.ones((3,3)))

seg2_opened = nd.binary_opening(seg2, np.ones((3,3)))
seg2_closed = nd.binary_closing(seg2_opened, np.ones((3,3)))

seg3_opened = nd.binary_opening(seg3, np.ones((3,3)))
seg3_closed = nd.binary_closing(seg3_opened, np.ones((3,3)))

all_seg_cleaned = np.zeros((denoise_ubyte.shape[0], denoise_ubyte.shape[1], 3))

all_seg_cleaned[seg1_closed] = (1,0,0)
all_seg_cleaned[seg2_closed] = (0,1,0)
all_seg_cleaned[seg3_closed] = (0,0,1)

plt.imshow(all_seg_cleaned)