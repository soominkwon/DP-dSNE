#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:22:07 2020

@author: soominkwon
"""

import numpy as np
from sklearn.utils import shuffle


def generate_data(mean, cov, sample_size, shuf=True):
    """ This function generates a set of (X, y) 2-D synthetic data drawn from a Gaussian
    distribution. 
    
    Arguments:
        mean: List of means for the negative and positive class, respectively
        cov: List of convariance matrices for the negative and positive class, respectively
        sample_size: Sample size for synthetic dataset. Generates sample_size/2 for negative and positive class
        shuf: If True, shuffles data
        
    Returns:
        X: Dataset of negative and positive classes centered around given means
        y: Labels corresponding to X
    
    """
    
    pos_samples = int(np.round(sample_size/2))
    neg_samples = int(np.round(sample_size/2))
    
    neg_x, neg_y = np.random.multivariate_normal(mean[0], cov[0], neg_samples).T
    pos_x, pos_y = np.random.multivariate_normal(mean[1], cov[1], pos_samples).T
    
    pos_x = np.expand_dims(pos_x, axis=1)
    neg_x = np.expand_dims(neg_x, axis=1)
    pos_y = np.expand_dims(pos_y, axis=1)
    neg_y = np.expand_dims(neg_y, axis=1)
    
    data_x = np.concatenate([pos_x, neg_x], axis=0)
    data_y = np.concatenate([pos_y, neg_y], axis=0)
    
    X = np.concatenate([data_x, data_y], axis=1)
    y = [-1]*neg_samples + [1]*pos_samples
    
    if shuf:
        X, y = shuffle(X, y)
        return X, y
    else:
        return X, y
    