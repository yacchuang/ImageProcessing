# -*- coding: utf-8 -*-
"""
Created on Tue May  9 08:36:54 2023

@author: Ya-Chen.Chuang
"""

import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt 
import os

###############################################################################
## Individual file
###############################################################################

image_path = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg/images/SP-014037_531426_Hs-APP-E9E10_05.tif"
mask_path = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg/masks/SP-014037_531426_Hs-APP-E9E10_label05.tif"

df = pd.DataFrame()

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
img2 = img.reshape(-1)
df['PixelValue'] = img2

msk = cv2.imread(mask_path)
msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
plt.imshow(msk)     
msk2 = msk.reshape(-1)
df['MaskValue'] = msk2          


df_mask = df['MaskValue']
df_img = df['PixelValue']
NewImg = []

row = 0
for index in df_mask: 
    if index == 76:
        NewImg.append(df_img[row])
    else:
        NewImg.append("0")
    
    row += 1

df['NewImage'] = NewImg

ImageArray = np.array(df['NewImage'].array)
Image = ImageArray.astype(np.uint8)
NewImg = Image.reshape(img.shape)

plt.imshow(NewImg)
plt.imsave('NeuN-Img/'+ 'SP-014037_531426_Hs-APP-E9E10_NeuN-_05.jpg', NewImg)



###############################################################################
## loop throught the whole folder
###############################################################################

image_path = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg/images/"
mask_path = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg/masks/"



for image_file in os.listdir(image_path):
    for mask_file in os.listdir(mask_path):
        
        print(mask_file)
        df = pd.DataFrame()
        img = cv2.imread(image_path + image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = img.reshape(-1)
        df['PixelValue'] = img2
    
        msk = cv2.imread(mask_path + mask_file)
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        msk2 = msk.reshape(-1)
        df['MaskValue'] = msk2    
        
        df_mask = df['MaskValue']
        df_img = df['PixelValue']
        NewImg = []
        
        row = 0
        for index in df_mask: 
            if index == 29:
                NewImg.append(df_img[row])
            else:
                NewImg.append("0")
            
            row += 1

        df['NewImage'] = NewImg
        
        ImageArray = np.array(df['NewImage'].array)
        Image = ImageArray.astype(np.uint8)
        NewImg = Image.reshape(img.shape)
        name = image_file.split("SP-")
        plt.imsave('NeuN+Img/'+ name[-1] +'.jpg', NewImg)
        
        
        
    
    

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


    

                  