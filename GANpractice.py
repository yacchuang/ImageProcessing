# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:30:24 2023

@author: Ya-Chen.Chuang
"""
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib as plt
import numpy as np

#Define input image dimensions
#Large images take too much time and resources.
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

##########################################################################
### Define Generator
# Given input a latent vector to generate images
# Given input of noise (latent) vector, the Generator produces an image.
def build_generator():
    
    noise_shape = (100,)  #1D array of size 100 (latent vector / noise)
    
    model = Sequential()
    
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization(momentum=0.8))
    
    model.summary()
    
    noise = Input(shape=noise_shape)
    img = model(noise) #Generated image
    
    return Model(noise, img)

#Alpha — α is a hyperparameter which controls the underlying value to which the
#function saturates negatives network inputs.
#Momentum — Speed up the training


##########################################################################
### Define descriminator
# Binary classification - sigmoid activation

def build_descriminator():
    
    model = Sequential()
    
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    
    img = Input(shape=img_shape)
    validity= model(img)
    
    return Model(img, validity)
#The validity is the Discriminator’s guess of input being real or not.



### Define a combined Gen and Des model, while discriminator is not trainable but only update weights for generator

def train(epochs, batch_size=128, save_interval=50):
    
    (X_train, _), (_, _) = mnist.load_data()
    
    # Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
    X_train = (X_train.astype(np.float32)-127.5)/127.5
    
    #Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    X_train = np.expand_dims(X_train, axis=3)
    
    half_batch = int(batch_size/2)
    
   
    
    for epoch in range(epochs):
        
        ## Train the descriminator first 
        # Select a random half batch of real images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]
        
        noise = np.random.normal(0,1,(half_batch, 100))
        
        # Generate a half batch of fake images
        gen_imgs = generator.predict(noise)
        
        # Train the discriminator on real and fake images, separately
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch,1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch,1)))
        
        d_loss = np.add(d_loss_real, d_loss_fake)/2
        
        ## Train the generator
        # Generate n_samples latent points as generator input to create fake images
        noise = np.random.normal(0,1(batch_size, 100))
        
        # The generator wants the discriminator to label the generated samples as valid (ones)
        valid_y = np.array([1]*batch_size)
        
        # Train the generator with noise as x and 1 as y
        g_loss = combined.train_on_batch(noise, valid_y)
        
        # Track the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    
        # If at save interval => save generated image samples
        if epoch % save_interval ==0:
            save_imgs(epoch)
            
        
       
def save_imgs(epoch):
    
    r, c = 5, 5
    noise = np.random.normal(0,1, (r*c, 100))
    gen_imgs = generator.predict(noise)
    
    # Rescale images 0-1
    gen_imgs = 0.5*gen_imgs +0.5
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap = "gray")
            axs[i,j].axis('off')
            cnt += 1
            
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()
    
##############################################################################

optimizer = Adam(0.0002, 0.5)  #Learning rate and momentum.   
    
        
        
        

### Generate fake images




# Generate fake images using above latent_dim and n_samples as input





### Train descriminator and generator

## enumerate through epoches and batches
# Take real images from datasets
# calculate loss from real images

# Take fake images from generator model
# calculate loss from fake images

# Take latent points to generate images



