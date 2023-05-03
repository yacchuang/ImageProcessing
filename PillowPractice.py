# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:23:29 2023

@author: Ya-Chen.Chuang
"""

#Reading images
from PIL import Image 


img = Image.open("C:/Users/ya-chen.chuang/Documents/QuPath/validation/QuPathValidationImages/SP-014200_1501-1F-CS-DRG1_EGFP-C1_mCherry-C2_MfaRBFOX3-C3_MfaGFAP-C4_yrwg_RGB.tiff.tif") #Not a numpy array

# Resize images
print(img.size)
img.show()

img.thumbnail((200,300))
img.save("test.tiff")
print(img.size)

small = img.resize((160, 150))

# show Images on external default viewer. This can be paint or photo viewer on Windows
small.show() 

# Cropping images
cropped = img.crop((500,500,1000,1000))
print(cropped.size)
cropped.show()

img_copy = img.copy()
img_copy.paste(cropped, (50,50))
img_copy.show()

# Color transforms, convert images between L (greyscale), RGB and CMYK
from PIL import Image 
img = Image.open("images/test_image.jpg")

grey_img = img.convert('L')  #L is for grey scale
grey_img.save("images/grey_img.jpg")

#Here is a way to automate image processing for multiple images.
from PIL import Image
import glob

path = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg/images/*.*"

for file in glob.glob(path):
    print(file)
    
    a = Image.open(file)
    rotate45 = a.rotate(45, expand = True)
    rotate45.show()