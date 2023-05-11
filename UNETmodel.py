# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:29:38 2023

@author: Ya-Chen.Chuang
"""

import tensorflow as tf
import os
import numpy as np
import random

from tqdm import tqdm      # shows the progress within for loop

from skimage.io import imread, imshow
from skimage.transform import resize
from matplotlib import pyplot as plt


# define a seed = np.random.seed to make share every run keeps the same number
seed = 42
np.random.seed = seed

# Define training and testing path
IMAGE_PATH = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/Cy5CellSegTraining/image/"
MASK_PATH = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/Cy5CellSegTraining/masks/"
TEST_PATH = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/Cy5CellSegTraining/test/"
PATH = "C:/Users/ya-chen.chuang/Documents/QuPath/MLtraining/Cy5CellSegTraining/"
# read all images IDs use os.walk
# TRAIN_ID = next(os.walk(TRAIN_PATH))[1]
# TEST_ID = next(os.walk(TEST_PATH))[1]

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 1280
IMG_CHANNEL = 3

# Create an empty array for training sets, filled in width, height, channels, dtype
X_train = np.zeros((len(os.listdir(IMAGE_PATH)), IMAGE_HEIGHT, IMAGE_WIDTH, IMG_CHANNEL))
Y_train = np.zeros((len(os.listdir(MASK_PATH)), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype = np.bool)

# resize training images and masks
n = 0
for file in os.listdir(IMAGE_PATH):
    img = imread(IMAGE_PATH + file)[:,:,:IMG_CHANNEL]
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode = 'constant', preserve_range=True)
    X_train[n] = img
    n += 1

m = 0
for file in os.listdir(MASK_PATH):
    msk = imread(MASK_PATH + file)[:,:]
    msk = resize(msk, (IMAGE_HEIGHT, IMAGE_WIDTH,1), mode = 'constant', preserve_range=True)
    Y_train[m] = msk
    m += 1
    

# resize test images
X_test = np.zeros((len(os.listdir(TEST_PATH)), IMAGE_HEIGHT, IMAGE_WIDTH, IMG_CHANNEL))
r = 0
for file in os.listdir(TEST_PATH):
    tst = imread(TEST_PATH + file)[:,:,:IMG_CHANNEL]
    tst = resize(tst, (IMAGE_HEIGHT, IMAGE_WIDTH), mode = 'constant', preserve_range=True)
    X_test[r] = tst
    r += 1

'''
image_x = random.randint(0, len(os.listdir(IMAGE_PATH))
plt.imshow(X_train[image_x,:,:,0])
plt.imshow(np.squeeze(Y_train[image_x]))
'''

# Build the model
inputs = tf.keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, IMG_CHANNEL))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

# Contraction path
c1 = tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1) # dropout
c1 = tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1) # Maxpool

c2 = tf.keras.layers.Conv2D(32, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.2)(c2) # dropout
c2 = tf.keras.layers.Conv2D(32, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2) # Maxpool

c3 = tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3) # dropout
c3 = tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3) # Maxpool

c4 = tf.keras.layers.Conv2D(128, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4) # dropout
c4 = tf.keras.layers.Conv2D(128, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4) # Maxpool

c5 = tf.keras.layers.Conv2D(256, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5) # dropout
c5 = tf.keras.layers.Conv2D(256, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c5)


#Expansive path 
U6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2, 2), padding='same')(c5)
U6 = tf.keras.layers.concatenate([U6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(U6)
c6 = tf.keras.layers.Dropout(0.2)(c6) # dropout
c6 = tf.keras.layers.Conv2D(128, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c6)

U7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2, 2), padding='same')(c6)
U7 = tf.keras.layers.concatenate([U7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(U7)
c7 = tf.keras.layers.Dropout(0.2)(c7) # dropout
c7 = tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c7)

U8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2, 2), padding='same')(c7)
U8 = tf.keras.layers.concatenate([U8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(U8)
c8 = tf.keras.layers.Dropout(0.1)(c8) # dropout
c8 = tf.keras.layers.Conv2D(32, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c8)

U9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2, 2), padding='same')(c8)
U9 = tf.keras.layers.concatenate([U9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(U9)
c9 = tf.keras.layers.Dropout(0.1)(c9) # dropout
c9 = tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation="sigmoid")(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# model.save("SomaSeg.h5")

###############################################################################
# Model checkpoint

checkpoint = tf.keras.callbacks.ModelCheckpoint('SomaSeg.h5', verbose=1, save_best_only=True)
callbacks = [tf.keras.callbacks.TensorBoard(log_dir='logs',update_freq='epoch')]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size = 16, epochs=50, callbacks=callbacks)
model.save("SomaSeg.h5")

# show the fitting result
idx = random.randint(0, len(X_train))

pred_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
pred_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
pred_test = model.predict(X_test, verbose=1)

pred_train_t = (pred_train > 0.5).astype(np.uint8)
pred_val_t = (pred_val > 0.5).astype(np.uint8)
pred_test_t = (pred_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
ix = random.randint(0, len(pred_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(pred_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(pred_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(pred_val_t[ix]))
plt.show()