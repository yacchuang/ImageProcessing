<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:20:11 2023

@author: Ya-Chen.Chuang
"""

import matplotlib.pyplot as plt
from skimage import io, img_as_float
import numpy as np
from tifffile import imsave
import cv2


img = img_as_float(io.imread("R:/YaChen/TrainingImage/DotImg-2/C2-SP-013812_RN14_SR-siRNA-Mfa-APP-O1-S1_Mfa-LOC101926433-C2_Mfa-RBFOX3-C3_Mfa-OLIG2-C4_wrgy_03.tif_cropped.tif"))
plt.imshow(img)
plt.hist(img.flat, bins=100, range=(0, 1)) 

# Very noisy image so histogram looks horrible. Let us denoise and see if it helps.

from skimage.restoration import denoise_nl_means, estimate_sigma

sigma_est = np.mean(estimate_sigma(img, multichannel=False))
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, 
                               patch_size=5, patch_distance=3, multichannel=False)
                           
#plt.hist(denoise_img.flat, bins=100, range=(0, 1)) 
# Much better histogram and now we can see two separate peaks. 
#Still close enough so cannot use histogram based segmentation.
#Let us see if we can get any better by some preprocessing.
#Let's try histogram equalization

from skimage import exposure   #Contains functions for hist. equalization

#eq_img = exposure.equalize_hist(denoise_img)
eq_img = exposure.equalize_adapthist(denoise_img)
#plt.imshow(eq_img, cmap='gray')
#plt.hist(denoise_img.flat, bins=100, range=(0., 1))

#Not any better. Let us stretch the hoistogram between 0.7 and 0.95

# The range of the binary image spans over (0, 1).
# For markers, let us include all between each peak.
markers = np.zeros(img.shape, dtype=np.uint)

markers[(eq_img < 0.4) & (eq_img > 0)] = 1
markers[(eq_img > 0.4) & (eq_img < 1)] = 2

from skimage.segmentation import random_walker
from skimage.color import rgb2gray

# Run random walker algorithm
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.random_walker
labels = random_walker(eq_img, markers, beta=10, mode='bf')

segm1 = (labels == 1)
segm2 = (labels == 2)
all_segments = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) #nothing but denoise img size but blank

all_segments[segm1] = (0,0,0)
all_segments[segm2] = (1,1,1)

all_segments = rgb2gray(all_segments)

plt.imshow(all_segments)
imsave("C5-SP-014200_2501-2F-CS-DRG1_EGFP-C1_mCherry-C2_MfaRBFOX3-C3_MfaGFAP-C4_yrwg_05_RW_0-7.tif", all_segments)

from scipy import ndimage as nd

segm1_closed = nd.binary_closing(segm1, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2, np.ones((3,3)))

all_segments_cleaned = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) 

all_segments_cleaned[segm1_closed] = (0,0,0)
all_segments_cleaned[segm2_closed] = (1,1,1)

plt.imshow(all_segments_cleaned) 
=======
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:20:11 2023

@author: Ya-Chen.Chuang
"""

import matplotlib.pyplot as plt
from skimage import io, img_as_float
import numpy as np
from tifffile import imsave


img = img_as_float(io.imread("R:/YaChen/TrainingImage/DotImg/C2-SP-013908_6001 10_SR-ASO-DMD5-S1_Mfa-PECAM1-C2_Mfa-S100b-C3_Mfa-GAP43-C4_wygr_02.tif"))

plt.hist(img.flat, bins=100, range=(0, 1)) 

# Very noisy image so histogram looks horrible. Let us denoise and see if it helps.

from skimage.restoration import denoise_nl_means, estimate_sigma

sigma_est = np.mean(estimate_sigma(img, multichannel=False))
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, 
                               patch_size=5, patch_distance=3, multichannel=False)
                           
#plt.hist(denoise_img.flat, bins=100, range=(0, 1)) 
# Much better histogram and now we can see two separate peaks. 
#Still close enough so cannot use histogram based segmentation.
#Let us see if we can get any better by some preprocessing.
#Let's try histogram equalization

from skimage import exposure   #Contains functions for hist. equalization

#eq_img = exposure.equalize_hist(denoise_img)
eq_img = exposure.equalize_adapthist(denoise_img)
#plt.imshow(eq_img, cmap='gray')
#plt.hist(denoise_img.flat, bins=100, range=(0., 1))

#Not any better. Let us stretch the hoistogram between 0.7 and 0.95

# The range of the binary image spans over (0, 1).
# For markers, let us include all between each peak.
markers = np.zeros(img.shape, dtype=np.uint)

markers[(eq_img < 0.7) & (eq_img > 0)] = 1
markers[(eq_img > 0.7) & (eq_img < 1)] = 2

from skimage.segmentation import random_walker
from skimage.color import rgb2gray

# Run random walker algorithm
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.random_walker
labels = random_walker(eq_img, markers, beta=10, mode='bf')

segm1 = (labels == 1)
segm2 = (labels == 2)
all_segments = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) #nothing but denoise img size but blank

all_segments[segm1] = (0,0,0)
all_segments[segm2] = (1,1,1)

all_segments = rgb2gray(all_segments)

plt.imshow(all_segments)
imsave("C2-SP-013908_6001 10_SR-ASO-DMD5-S1_Mfa-PECAM1-C2_Mfa-S100b-C3_Mfa-GAP43-C4_wygr_RW_0-7_02.tif", all_segments)

from scipy import ndimage as nd

segm1_closed = nd.binary_closing(segm1, np.ones((3,3)))
segm2_closed = nd.binary_closing(segm2, np.ones((3,3)))

all_segments_cleaned = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) 

all_segments_cleaned[segm1_closed] = (0,0,0)
all_segments_cleaned[segm2_closed] = (1,1,1)

plt.imshow(all_segments_cleaned) 
>>>>>>> fc4e981e5ff98b5f7d5eab3dca0d774ed90476fa
imsave("C4-Open3-2ndExp_Scan1.qptiff-1_RW_0-7_cleaned.tif", all_segments_cleaned)