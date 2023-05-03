# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:30:24 2023

@author: Ya-Chen.Chuang
"""

### Define descriminator
# Binary classification - sigmoid activation


### Define Generator
# Given input a latent vector to generate images

### Define a combined Gen and Des model, while discriminator is not trainable but only update weights for generator

### Generate fake images
# Take half batch of real images to train GAN, assign label 1 for real and 0 for fake

# Generate n_samples latent points as generator input to create fake images

# Generate fake images using above latent_dim and n_samples as input





### Train descriminator and generator

## enumerate through epoches and batches
# Take real images from datasets
# calculate loss from real images

# Take fake images from generator model
# calculate loss from fake images

# Take latent points to generate images



