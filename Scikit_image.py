# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:45:34 2023

@author: Ya-Chen.Chuang
"""

from skimage import io

from matplotlib import pyplot as plt

img = io.imread("C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/HandEcompressed_Scan1.qptiff", as_gray=True)

from skimage.transform import rescale, resize, downscale_local_mean

#Rescale, resize image by a given factor. While rescaling image
#gaussian smoothing can performed to avoid anti aliasing artifacts.
rescaled_img = rescale(img, 1.0/4.0, anti_aliasing=True)

#Resize, resize image to given dimensions (shape)
resized_img = resize(img, (200,200))

downscaled_img = downscale_local_mean(img, (4,3))

plt.imshow(resized_img)


###############################
# Edge Detection

from skimage.filters import roberts, sobel, scharr, prewitt

img = io.imread("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg/images/SP-014037_531426_Hs-APP-E9E10_01.tif", as_gray=True)

edge = prewitt(img)
plt.imshow(edge, cmap = 'gray')

#Another edge filter is Canny. This is not just a single operation
#It does noise reduction, gradient calculation, and edge tracking among other things. 
#Canny creates a binary file, true or false pixels. 

from skimage.feature import canny

edge_canny = canny(img, sigma = 5)
plt.imshow(edge_canny)


###############################################
#Image deconvolution
#Uses deconvolution to sharpen images. 

from skimage import restoration

import numpy as np

psf = np.ones((2,2))/4
deconvolved, _ = restoration.unsupervised_wiener(img, psf)
plt.imshow(deconvolved) 



#########################################
#Let's find a way to calculate the area of scratch in would healing assay
#Entropy filter
#e.g. scratch assay where you have rough region with cells and flat region of scratch.
#entropy filter can be used to separate these regions

import matplotlib.pyplot as plt

from skimage import io, restoration

from skimage.filters.rank import entropy

from skimage.morphology import disk

img = io.imread("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg/images/SP-014037_531426_Hs-APP-E9E10_01.tif")
plt.imshow(img)
entropy_img = entropy(img, disk(10))
plt.imshow(entropy_img)

#Once you have the entropy iamge you can apply a threshold to segment the image
#If you're not sure which threshold works fine, skimage has a way for you to check all 


from skimage.filters import try_all_threshold

fig, ax = try_all_threshold(entropy_img, figsize = (10, 8), verbose = False)
plt.show()


from skimage.filters import threshold_otsu

thresh = threshold_otsu(entropy_img)

binary = entropy_img >= thresh

plt.imshow(binary)