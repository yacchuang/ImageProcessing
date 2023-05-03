# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 08:12:02 2023

@author: Ya-Chen.Chuang
"""

import cv2

## Read Image
image = cv2.imread("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg/masks/SP-014037_531426_Hs-APP-E9E10_label01.tif",1)

print(image.shape)
print("center", image[300, 700])

mask = image[:, :, 2]

cv2.imshow("example", mask)
cv2.waitKey(10)
cv2.imwrite("SP-014037_531426_Hs-APP-E9E10_NeuN+_01.tif.tif", mask)
# cv2.destroyAllWindows()

## Split Clannels
b, r, g = cv2.split(image)

cv2.imshow("blue", b)
cv2.imshow("red", r)
cv2.imshow("green", g)
cv2.waitKey(10)

## Merge Channels
image_merge = cv2.merge((b, g, r))


## Resize Images
resized = cv2.resize(image, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
cv2.imshow("original", image)
cv2.imshow("resized", resized)
cv2.waitKey(10)
#cv2.destroyAllWindows()

'''
image = cv2.imread("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/Cy5CellSegTraining/images_original/SP-014200_1501-1F-CS-DRG1_EGFP-C1_mCherry-C2_MfaRBFOX3-C3_MfaGFAP-C4_yrwg_01.tif",1)


cv2.imshow("original", image)
b, r, g = cv2.split(image)
image_merge = cv2.merge((b, g, r))
cv2.imshow("stack", image_merge)
'''