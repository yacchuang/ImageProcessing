# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:20:11 2023

@author: Ya-Chen.Chuang
"""

import matplotlib.pyplot as plt
from skimage import io, img_as_float
import numpy as np
from tifffile import imsave
import os


image_path = "R:/YaChen/TrainingImage/DotImg/"
mask_path = "R:/YaChen/TrainingImage/DotMask/"
for image in os.listdir(image_path):
    print(image)
    
    img = img_as_float(io.imread(image_path + image))
    plt.hist(img.flat, bins=100, range=(0, 1)) 
                       

    from skimage.restoration import denoise_nl_means, estimate_sigma

    sigma_est = np.mean(estimate_sigma(img, multichannel=False))
    denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, 
                               patch_size=5, patch_distance=3, multichannel=False)
                           


    from skimage import exposure   #Contains functions for hist. equalization

    #eq_img = exposure.equalize_hist(denoise_img)
    eq_img = exposure.equalize_adapthist(denoise_img)
    #plt.imshow(eq_img, cmap='gray')
    #plt.hist(denoise_img.flat, bins=100, range=(0., 1))


    # The range of the binary image spans over (0, 1).

    markers = np.zeros(img.shape, dtype=np.uint)

    markers[(eq_img < 0.7) & (eq_img > 0)] = 1
    markers[(eq_img > 0.7) & (eq_img < 1)] = 2

    from skimage.segmentation import random_walker
    from skimage.color import rgb2gray


    labels = random_walker(eq_img, markers, beta=10, mode='bf')

    segm1 = (labels == 1)
    segm2 = (labels == 2)
    all_segments = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) #nothing but denoise img size but blank

    all_segments[segm1] = (0,0,0)
    all_segments[segm2] = (1,1,1)

    all_segments = rgb2gray(all_segments)

    plt.imshow(all_segments)
    imsave(mask_path + image + "_RW_0-7.tif", all_segments)

'''
from scipy import ndimage as nd

segm1_closed = nd.binary_closing(segm1, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2, np.ones((3,3)))

all_segments_cleaned = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) 

all_segments_cleaned[segm1_closed] = (0,0,0)
all_segments_cleaned[segm2_closed] = (1,1,1)

plt.imshow(all_segments_cleaned) 
imsave("C4-Open3-2ndExp_Scan1.qptiff-1_RW_0-7_cleaned.tif", all_segments_cleaned)
'''