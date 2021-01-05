#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:36:04 2021

@author: soominkwon
"""

import numpy as np
import pylab
from adaclip_tsne import noisy_tsne
from generate_synth_data import generate_data

# initializing parameters
var = 1e-3
iterations = 500

# parameters for data
mean_list1 = [[-5, -5], [5, 5]]
mean_list2 = [[-5, 5], [5, -5]]
cov_list = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]

# generating data
X_1, y_1 = generate_data(mean=mean_list1, cov=cov_list, sample_size=200, shuf=True)
X_2, y_2 = generate_data(mean=mean_list2, cov=cov_list, sample_size=200, shuf=True)    
X = np.concatenate([X_1, X_2], axis=0)
y = np.concatenate([y_1, y_2], axis=0)
    

# running t-SNE
Y, dY = noisy_tsne(X=X, noise_variance=var, iters=iterations, no_dims=2, 
                                perplexity=40.0, PCA=False, initial_dims=50, grad_clip=True)
    
pylab.scatter(X[:, 0], X[:, 1], 20, y)
pylab.title('Original Plot')
pylab.show()
    
pylab.scatter(Y[:, 0], Y[:, 1], 20, y)
pylab.title('DP t-SNE Plot')
pylab.show()