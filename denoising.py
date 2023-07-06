# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:47:38 2023

@author: Ya-Chen.Chuang
"""

from skimage import io, img_as_float
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np

img = io.imread("R:/YaChen/FL_miRNA/C2-SP-013908_9001 10_SR-ASO-DMD5-S1_Mfa-PECAM1-C2_Mfa-S100b-C3_Mfa-GAP43-C4_wygr.ome.tiff.tif")
gaussian_img = nd.gaussian_filter(img, sigma =15)

plt.imshow(gaussian_img)
plt.imsave("R:/YaChen/FL_miRNA/test/C2-SP-013908_9001 10_gaussian-15.jpg", gaussian_img, cmap = 'jet')

med_img = nd.median_filter(img, size = 2)

plt.imshow(med_img)

from filterpy.kalman import KalmanFilter
kf = KalmanFilter(dim_x=3, dim_z=1)



img = img_as_float("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/Cy5CellSegTraining/image/SP-014200_1501-1F-CS-DRG1_EGFP-C1_mCherry-C2_MfaRBFOX3-C3_MfaGFAP-C4_yrwg_01.tif")
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise = denoise_nl_means(img, h=2*sigma_est, fast_mode=False, patch_size = 5, patch_distance=3, multichannel= True)
plt.imshow(denoise)