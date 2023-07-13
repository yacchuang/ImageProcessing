<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:07:30 2023

@author: Ya-Chen.Chuang
"""

import cv2
from matplotlib import pyplot as plt 
from tifffile import imsave
import os

image_path = "C:/Users/ya-chen.chuang/Documents/QuPath/ProbeSeg/ClusterImg/New folder/"
cropped_path = "C:/Users/ya-chen.chuang/Documents/QuPath/ProbeSeg/CroppedImg/"


for image in os.listdir(image_path):
    print(image)
    
    img = cv2.imread(image_path + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img)

    y=0
    x=0
    h=512
    w=512
    crop_image = img[x:w, y:h]
    
    plt.imshow(crop_image)
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:07:30 2023

@author: Ya-Chen.Chuang
"""

import cv2
from matplotlib import pyplot as plt 
from tifffile import imsave
import os

image_path = "C:/Users/ya-chen.chuang/Documents/QuPath/ProbeSeg/images-new/"
cropped_path = "C:/Users/ya-chen.chuang/Documents/QuPath/ProbeSeg/CroppedImg/"


for image in os.listdir(image_path):
    print(image)
    
    img = cv2.imread(image_path + image)
    plt.imshow(img)

    y=0
    x=0
    h=512
    w=512
    crop_image = img[x:w, y:h]
    
    plt.imshow(crop_image)
>>>>>>> fc4e981e5ff98b5f7d5eab3dca0d774ed90476fa
    imsave(cropped_path + image + "_cropped.tif", crop_image)