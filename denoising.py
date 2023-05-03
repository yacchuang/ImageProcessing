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

img = io.imread("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/Cy5CellSegTraining/image/SP-014200_1501-1F-CS-DRG1_EGFP-C1_mCherry-C2_MfaRBFOX3-C3_MfaGFAP-C4_yrwg_01.tif")
gaussian_img = nd.gaussian_filter(img, sigma =3)

plt.imshow(gaussian_img)


med_img = nd.median_filter(img, size = 2)

plt.imshow(med_img)


img = img_as_float("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/Cy5CellSegTraining/image/SP-014200_1501-1F-CS-DRG1_EGFP-C1_mCherry-C2_MfaRBFOX3-C3_MfaGFAP-C4_yrwg_01.tif")
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise = denoise_nl_means(img, h=2*sigma_est, fast_mode=False, patch_size = 5, patch_distance=3, multichannel= True)
plt.imshow(denoise)