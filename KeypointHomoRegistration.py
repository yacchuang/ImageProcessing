# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:40:10 2023

@author: Ya-Chen.Chuang
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1_path = "C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/QuPathValidationImages/SP-013812_RN14_SR-siRNA-Mfa-APP-S1-O1_Mfa-LOC101926433-C2_Mfa-PECAM1-C3_wrg.tif";
img2_path = "C:/Users/ya-chen.chuang/Documents/QuPath/SampleImages/QuPathValidationImages/SP-013812_RN14_SR-siRNA-Mfa-APP-O1-S1_Mfa-LOC101926433-C2_Mfa-RBFOX3-C3_Mfa-OLIG2-C4_wrgy.tif"

# read 2 images using cv2.imread
im1 = cv2.imread(img1_path)
im2 = cv2.imread(img2_path)

img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create(50000)  #Registration works with at least 50 points


# find the keypoints and descriptors with orb
kp1, des1 = orb.detectAndCompute(img1, None)  #kp1 --> list of keypoints
kp2, des2 = orb.detectAndCompute(img2, None)

# create Matcher object
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

# Match descriptors.
matches = matcher.match(des1, des2, None)  #Creates a list of all matches, just like keypoints


# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 20 matches
img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None)

cv2.imshow("Matches image", img3)
cv2.waitKey(0)
## Register 2 images using keypoints

## Extract location of good matches using RANSAC
## RANSAC needs all key points indexed, first set indexed to queryIdx
## Second set to #trainIdx. 

points1 = np.zeros((len(matches), 2), dtype = np.float32)  #Prints empty array of size equal to (matches, 2)
points2 = np.zeros((len(matches), 2), dtype = np.float32)

for i, match in enumerate(matches):
    points1[i,:] = kp1[match.queryIdx].pt   #gives index of the descriptor in the list of query descriptors
    points2[i,:] = kp2[match.queryIdx].pt   #gives index of the descriptor in the list of query descriptors


# Find homography

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# Use homography

height, width, channels = im1.shape
im2Reg = cv2.warpPerspective(im2, h, (width, height))

print("Estimated homography : \n",  h)

cv2.imshow("Registered image", im2Reg)
cv2.waitKey()