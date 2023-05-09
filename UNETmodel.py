# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:29:38 2023

@author: Ya-Chen.Chuang
"""

import tensorflow as tf

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMG_CHANNEL = 3



inputs = tf.keras.layers.Input((IMAGE_HEIGHT, IMAGE_WIDTH, IMG_CHANNEL))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

#Contraction path
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