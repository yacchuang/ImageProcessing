# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:01:40 2023

@author: Ya-Chen.Chuang
"""



import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt 
import os

###############################################################################
## Read training images and extract features
###############################################################################

# create an empty dataframe
image_dataset = pd.DataFrame()

# read all images in the folder
image_path = "R:/YaChen/TrainingImage/TradMLSeg/images/"
for image in os.listdir(image_path):
    print(image)
    df1 = pd.DataFrame()
    img = cv2.imread(image_path + image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    img2 = img.reshape(-1)
    
###############################################################################
## Add data to the dataframe
    # df1['Image name'] = str(image)
    df1['Pixel Value'] = img2
    
###############################################################################
## Generate All features
# Gabor features
    num = 1

    for sigma in (1,3):
        for theta in range(4):
            theta = theta/ 8. * np.pi
            for lamda in np.arange(0, np.pi, np.pi / 4): 
                for gamma in (0.05, 0.5, 0.95):
                    Gabor_label = "Gabor" + str(num)
                    
                    GaborFilter = cv2.getGaborKernel((50, 50), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                
                
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, GaborFilter)
                    df1[Gabor_label] = fimg
                    print(Gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                
                    num += 1

    # print(df1.head())

# Canny edge

    edges = cv2.Canny(img, 100, 200)
    edges1 = edges.reshape(-1) 
    # plt.imshow(edges)

    df1['Canny edge'] = edges1
    # print(df.head())

# Other filters
    from skimage.filters import sobel, roberts, scharr, prewitt

    sobel_img = sobel(img)
    sobel1 = sobel_img.reshape(-1)
    df1['Sobel'] = sobel1

    robert_img = roberts(img)
    robert1 = robert_img.reshape(-1)
    df1['roberts'] = robert1

    scharr_img = scharr(img)
    scharr1 = scharr_img.reshape(-1)
    df1['scharr'] = scharr1

    prewitt_img = prewitt(img)
    prewitt1 = prewitt_img.reshape(-1)
    df1['prewitt'] = prewitt1

# Gaussian filter
    from scipy import ndimage as nd
    gaussian_img3 = nd.gaussian_filter(img, sigma =3)
    gaussian3 = gaussian_img3.reshape(-1)
    df1['Gaussian3'] = gaussian3

    gaussian_img7 = nd.gaussian_filter(img, sigma =7)
    gaussian7 = gaussian_img7.reshape(-1)
    df1['Gaussian7'] = gaussian7
    
# Median filter
    med_img3 = nd.median_filter(img, size = 3)
    med3 = med_img3.reshape(-1)
    df1['median filter 3'] = med3

    med_img7 = nd.median_filter(img, size = 7)
    med7 = med_img7.reshape(-1)
    df1['median filter 7'] = med7

# Variance with size = 3
    var_img3 = nd.generic_filter(img, np.var, size = 3)
    var3 = var_img3.reshape(-1)
    df1['Variance 3'] = var3
    

    image_dataset = image_dataset.append(df1)
    
###############################################################################
## Read mask (label) images and create another dataframe
###############################################################################
    
mask_dataset = pd.DataFrame()
mask_path = "R:/YaChen/TrainingImage/TradMLSeg/masks/"

for mask in os.listdir(mask_path):
    df2 = pd.DataFrame()
    msk = cv2.imread(mask_path + mask)
    msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    msk2 = msk.reshape(-1)
    
    # df2['Label name'] = str(mask)
    df2['Label Value'] = msk2
    
    mask_dataset = mask_dataset.append(df2)
    
###############################################################################
## Combine image and mask dataframe, and ready to train RF or SVM
###############################################################################

dataset = pd.concat([image_dataset, mask_dataset], axis = 1) # axis = 1 meaning concatnate along column

# Split X and Y dataset, and split training and testing dataset
from sklearn.model_selection import train_test_split

Y = dataset['Label Value'].values
X = dataset.drop(labels = ['Label Value'], axis =1)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


###############################################################################
## Define a classifier and fit a model with training dataset
###############################################################################

from sklearn.svm import LinearSVC

SVMmodel = LinearSVC(max_iter=500000)
SVMmodel.fit(X_train, y_train)

y_pred = SVMmodel.predict(X_test)
# y_proba = SVMmodel.predict_proba(X_test)


from sklearn.ensemble import RandomForestClassifier

RFmodel = RandomForestClassifier(n_estimators=10, random_state=30)
RFmodel.fit(X_train, y_train)

RF_pred_train = RFmodel.predict(X_train)
RF_y_pred = RFmodel.predict(X_test)


###############################################################################
## Accuracy check
###############################################################################

from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(y_test, RF_y_pred))


###############################################################################
## Save the model for future use
###############################################################################

import pickle

filename = "FLProbeClusterSeg"
pickle.dump(RFmodel, open(filename, 'wb'))

Load_model = pickle.load(open(filename, 'rb'))
result = Load_model.predict(X)

Segment = result.reshape(img.shape)

plt.imshow(Segment)
plt.imsave('FLProbeClusterSeg.jpg', Segment, cmap='jet')

