# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:45:33 2023

@author: Ya-Chen.Chuang
"""

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

'''
img = cv2.imread("C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg-1/images/SP-014037_531426_Hs-APP-E9E10_1.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
'''

########################### Create Features ###################################
def feature_extraction(img):

    df = pd.DataFrame()

# Add original pixel values
    img2 = img.reshape(-1)
    df['Original Image'] = img2

## Add other features
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
                    df[Gabor_label] = fimg
                    print(Gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                
                    num += 1

    print(df.head())

# Canny edge

    edges = cv2.Canny(img, 100, 200)
    edges1 = edges.reshape(-1) 
    plt.imshow(edges)

    df['Canny edge'] = edges1
    # print(df.head())

# Other filters
    from skimage.filters import sobel, roberts, scharr, prewitt

    sobel_img = sobel(img)
    sobel1 = sobel_img.reshape(-1)
    df['Sobel'] = sobel1

    robert_img = roberts(img)
    robert1 = robert_img.reshape(-1)
    df['roberts'] = robert1

    scharr_img = scharr(img)
    scharr1 = scharr_img.reshape(-1)
    df['scharr'] = scharr1

    prewitt_img = prewitt(img)
    prewitt1 = prewitt_img.reshape(-1)
    df['prewitt'] = prewitt1

# Gaussian filter
    from scipy import ndimage as nd
    gaussian_img3 = nd.gaussian_filter(img, sigma =3)
    gaussian3 = gaussian_img3.reshape(-1)
    df['Gaussian3'] = gaussian3

    gaussian_img7 = nd.gaussian_filter(img, sigma =7)
    gaussian7 = gaussian_img7.reshape(-1)
    df['Gaussian7'] = gaussian7
    
# Median filter
    med_img3 = nd.median_filter(img, size = 3)
    med3 = med_img3.reshape(-1)
    df['median filter 3'] = med3

    med_img7 = nd.median_filter(img, size = 7)
    med7 = med_img7.reshape(-1)
    df['median filter 7'] = med7

# Variance with size = 3
    var_img3 = nd.generic_filter(img, np.var, size = 3)
    var3 = var_img3.reshape(-1)
    df['Variance 3'] = var3
    
    return df

'''
# Add label image
label = cv2.imread('C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg-1/masks/SP-014037_531426_Hs-APP-E9E10_1_label.tif')
label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
label2 = label.reshape(-1)

df['label'] = label2


# df.to_csv("MLfeatures.csv")

######################### Machine Learning ####################################
# separate dependent variables: label, and independent variables
Y = df['label'].values
X = df.drop(labels = ['label'], axis =1)

# Split train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

# Import ML algorithm and train the model
from sklearn.ensemble import RandomForestClassifier

RFmodel = RandomForestClassifier(n_estimators=10, random_state=30)
RFmodel.fit(X_train, y_train)

# test the prediction on training data first to verifty the model
pred_train = RFmodel.predict(X_train)
y_pred = RFmodel.predict(X_test)

from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(y_test, y_pred))

# Find out the importancy of each feature
importance = list(RFmodel.feature_importances_)
feature_list = list(X.columns)
feature_imp = pd.Series(RFmodel.feature_importances_, index = feature_list).sort_values(ascending = False)
print(feature_imp)

# Save and use the model
import pickle

filename = "NeunSeg"
pickle.dump(RFmodel, open(filename, 'wb'))

load_model = pickle.load(open(filename, 'rb'))
result = load_model.predict(X)

Segment = result.reshape(img.shape)

plt.imshow(Segment)
plt.imsave('NeunSeg1.jpg', Segment, cmap='jet')

'''
###############################################################################

import glob
import pickle

filename = "NeunSeg"
load_model = pickle.load(open(filename, 'rb'))

path = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/BF2ClassCellSeg-1/images/SP-014037_531426_Hs-APP-E9E10_?.tif"

for file in glob.glob(path):
    print(file)
    
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    X = feature_extraction(img)
    result = load_model.predict(X)
    
    Segment = result.reshape(img.shape)
    name = file.split("7_")
    plt.imsave('Segment/'+ name[-1]+'.jpg', Segment, cmap='jet')
    

