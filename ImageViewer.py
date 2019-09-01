# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 23:01:11 2019

@author: FN_Ne
"""
import numpy as np
import matplotlib.pyplot as plt 

with np.load('data/mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    
plt.imshow(training_images[5].reshape(28,28),cmap='gray')
plt.show()