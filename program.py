# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 22:40:40 2019

@author: FN_Ne
"""

import NNsigmoid1 as nn
import numpy as np


learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5


with np.load('data/mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']


layer_sizes = (784,5,10)


net = nn.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images,training_labels)
