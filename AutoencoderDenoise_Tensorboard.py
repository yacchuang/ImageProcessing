# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:23:09 2023

@author: Ya-Chen.Chuang
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

import numpy as np
from matplotlib import pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/ 255.
x_test = x_test.astype('float32')/ 255.       #conver images to float and normalize

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Add artificial noise
noise_factor = 0.5

x_train_noise = x_train + noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noise = x_test + noise_factor*np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip all values between 0-1
x_train_noise = np.clip(x_train_noise, 0., 1.)
x_test_noise = np.clip(x_test_noise, 0., 1.)

# Show some noise images
plt.figure(figsize=(20,2))
for i in range(1,10):
    ax = plt.subplot(1, 10, i)
    plt.imshow(x_test_noise[i].reshape(28,28), cmap="binary")
    
plt.show()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', input_shape = (28,28,1)))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))
model.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D((2,2), padding = 'same'))

model.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(1, (3, 3), activation = 'relu', padding = 'same'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()

# history = model.fit(x_train_noise, x_train, epochs=10, batch_size=64, shuffle=True, validation_data=(x_test_noise, x_test), callbacks=[tensorboard])
# model.evaluate(x_test_noise, x_test)

# model.save('AutoencodeDenoise.model')

# pred_img = model.predict(x_test_noise)

###########################################################
#Tensorboard callback

from tensorflow.keras.callbacks import TensorBoard

file_name = 'AutoencodeDenoise'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))



history = model.fit(x_train_noise, x_train, epochs=10, batch_size=64, shuffle=True, validation_data=(x_test_noise, x_test), callbacks=[tensorboard])
model.evaluate(x_test_noise, x_test)
# print("Accuracy = ", (acc * 100.0), "%")

model.save('AutoencodeDenoise.model')

pred_img = model.predict(x_test_noise)


############################################################################
## Open Tensorboard

# Open Anaconda Prompt

# activate tf1_model_export
# python -m tensorboard.main --logdir=logs/
# launch Tensorboard using the link provided
############################################################################

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test_noise[i].reshape(28, 28), cmap="binary")
    
    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 +i+ 1)
    plt.imshow(pred_img[i].reshape(28, 28), cmap="binary")

plt.show()


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(x_test_noise, pred_img)

import seaborn as sns
sns.heatmap(cm, annot=True)


