# -*- coding: utf-8 -*-
"""
Created on Tue May  9 08:36:54 2023

@author: Ya-Chen.Chuang
"""

import pandas as pd
import cv2
import numpy as np
from skimage import io, img_as_float
from matplotlib import pyplot as plt 
import os
from tifffile import imsave


###############################################################################
## Individual file
###############################################################################

image_path = "R:/YaChen/TrainingImage/DotImg/C2-SP-013908_9501_10_SR-ASO-DMD5-S1_Mfa-PECAM1-C2_Mfa-S100b-C3_Mfa-GAP43-C4_wygr_10.tif"
mask_path = "R:/YaChen/TrainingImage/DotMask/C2-SP-013908_9501_10_SR-ASO-DMD5-S1_Mfa-PECAM1-C2_Mfa-S100b-C3_Mfa-GAP43-C4_wygr_10.tif_RW_0-7.tif"
masked_img = "R:/YaChen/TrainingImage/MaskedImg/"

df = pd.DataFrame()

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
img2 = img.reshape(-1)
df['PixelValue'] = img2

# msk = cv2.imread(mask_path)
msk = img_as_float(io.imread(mask_path))
# msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
plt.imshow(msk)     
msk2 = msk.reshape(-1)
df['MaskValue'] = msk2          


df_mask = df['MaskValue']
df_img = df['PixelValue']
NewImg = []

row = 0
for index in df_mask: 
    if index == 1:
        NewImg.append(df_img[row])
    else:
        NewImg.append("0")
    
    row += 1

df['NewImage'] = NewImg

ImageArray = np.array(df['NewImage'].array)
Image = ImageArray.astype(np.uint8)
NewImg = Image.reshape(img.shape)

plt.imshow(NewImg)
imsave(masked_img + 'C2-SP-013908_9501_10_SR-ASO-DMD5-S1_Mfa-PECAM1-C2_Mfa-S100b-C3_Mfa-GAP43-C4_wygr_10_RW_0-7_MaskedImage.tif', NewImg)

'''

###############################################################################
## loop throught the whole folder
###############################################################################

image_path = "R:/YaChen/TrainingImage/DotImg/"
mask_path = "R:/YaChen/TrainingImage/DotMask/"
masked_img = "R:/YaChen/TrainingImage/MaskedImg/"



for image_file in os.listdir(image_path):
    for mask_file in os.listdir(mask_path):
        
        print(mask_file)
        df = pd.DataFrame()
        img = cv2.imread(image_path + image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = img.reshape(-1)
        df['PixelValue'] = img2
    
        # msk = cv2.imread(mask_path + mask_file)
        # msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        msk = img_as_float(io.imread(mask_path + mask_file))
        msk2 = msk.reshape(-1)
        df['MaskValue'] = msk2    
        
        df_mask = df['MaskValue']
        df_img = df['PixelValue']
        NewImg = []
        
        row = 0
        for index in df_mask: 
            if index == 1:
                NewImg.append(df_img[row])
            else:
                NewImg.append("0")
            
            row += 1

        df['NewImage'] = NewImg
        
        ImageArray = np.array(df['NewImage'].array)
        Image = ImageArray.astype(np.uint8)
        NewImg = Image.reshape(img.shape)
        # name = image_file.split("SP-")
        imsave(masked_img + image_file + '_MaskedImg.tif', NewImg)
        
        
        
'''    
    

'''
for index in df_mask:   
    if index == 255:                    # Bg
        NewMask.append("0")
        
    elif index == 76:                   # NeuN-
        NewMask.append("0")
        
    elif index == 29:                   # NeuN+
        NewMask.append("1")

    row += 1
'''    


    

                  