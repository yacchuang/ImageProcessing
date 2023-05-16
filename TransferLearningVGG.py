# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:15:20 2023

@author: Ya-Chen.Chuang
"""

from keras.layers import Conv2D, UpSampling2D, Input
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img

import numpy as np
import tensorflow as tf
import keras
import os


'''
Replace the encoder part with VGG16:
we don’t need it as a classifier, we need it as a feature extractor.
we will be using that latent space volume as a feature vector to be input to the decoder.
and the decoder is going to learn the mapping from the latent space vector to ab channels.
We use first 19 layers from VGG16 without changing its original weight to extract features.
'''

from keras.applications.vgg16 import VGG16
vggmodel = VGG16()
newmodel = Sequential() 

for i, layer in enumerate(vggmodel.layers):
    if i <19:
        newmodel.add(layer)
newmodel.summary()

for layer in newmodel.layers:
    layer.trainable = False
    
    
# load training images, and convert to LAB space
# extract "L channel" as X vector, grey images
# assign A and B channels to Y vector
# normalize Y from -127 ~ 128 to -1 ~1


# Extract VGG features
vggfeatures = []
for i, sample in enumerate(X):
  sample = gray2rgb(sample)
  sample = sample.reshape((1,224,224,3))
  prediction = newmodel.predict(sample)
  prediction = prediction.reshape((7,7,512))
  vggfeatures.append(prediction)
vggfeatures = np.array(vggfeatures)
print(vggfeatures.shape)

#Decoder
model = Sequential()

model.add(Conv2D(256, (3,3), activation='relu', padding='same', input_shape=(7,7,512)))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.summary()


model.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
model.fit(vggfeatures, Y, verbose=1, epochs=10, batch_size=128)

model.save('colorize_autoencoder_VGG16.model')

############################################
 