#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:07:41 2020

@author: soominkwon
"""

""" Working AdaCliP implementation. Refer to the original paper for more details.
"""

import numpy as np


def noise_addition(gradient_matrix, m_mat, b_mat, noise_variance):
    """ Function that adds noise given variables.

        Arguments:
            gradient_matrix: Matrix of all the gradients from gradient descent
            m_mat: Matrix of 'm' or mean values
            b_mat: Matrix of 'b' values from paper
            noise_variance: variance to add Gaussian nosie

    """
    noisy_gradient_mat = np.zeros((gradient_matrix.shape))
    new_gradient_matrix = (gradient_matrix - m_mat) / b_mat
    norm_mat = np.linalg.norm(new_gradient_matrix, axis=1)
    
    for i in range(gradient_matrix.shape[0]):
        epsilon = np.random.normal(0, noise_variance, size = [gradient_matrix.shape[1], ])
        clipped_grads = new_gradient_matrix[i, :] / max(1, norm_mat[i])
        clip_noisy_grads = clipped_grads + epsilon
        noisy_gradient_mat[i, :] = clip_noisy_grads
    
    # scaling gradient matrix back
    scaled_matrix = b_mat*noisy_gradient_mat + m_mat

    return scaled_matrix


def compute_b_matrix(s_mat):
    # computing B matrix
    b_mat = np.zeros((s_mat.shape))
    for i in range(b_mat.shape[0]):
        s_vec = s_mat[i, :]
        for j in range(b_mat.shape[1]):
            b_mat[i, j] = np.sqrt(s_vec[j]) * np.sqrt(np.sum(s_vec))
    return b_mat
    

def compute_s_matrix(s_mat, beta_2, variance_mat):
    # computing S matrix
    new_s_mat = beta_2*s_mat + ((1-beta_2)*variance_mat)
    return new_s_mat
    
    
def update_mean(gradient_matrix, mean_matrix, beta_1):
    first_term = mean_matrix * beta_1
    second_term = (1-beta_1)*gradient_matrix
    
    update_mean = first_term + second_term
    
    return update_mean


def update_variance(gradient_matrix, mean_matrix, b_matrix, noise_variance, h_1, h_2):
    first_term = (gradient_matrix - mean_matrix)**2
    second_term = (b_matrix)**2 * noise_variance
    
    max_term = np.maximum((first_term-second_term), h_1)
    variance = np.minimum(max_term, h_2)
    
    return variance


def AdaClip(gradient_matrix, noise_variance, m_mat, s_mat, beta_1, beta_2, h_1, h_2):
    """ Putting it all together for AdaCliP implementation. You should initialize your variables (e.g. s_mat, b_mat) outside
        of this function.
    
    
    """
    b_mat = compute_b_matrix(s_mat)
    noisy_gradient_matrix = noise_addition(gradient_matrix=gradient_matrix, m_mat=m_mat,
                                                b_mat=b_mat, noise_variance=noise_variance)
    m_mat = update_mean(gradient_matrix=noisy_gradient_matrix, mean_matrix=m_mat,
                                                beta_1=beta_1)
    var_mat = update_variance(gradient_matrix=noisy_gradient_matrix, mean_matrix=m_mat,
                                                 b_matrix=b_mat, noise_variance=noise_variance, h_1=h_1, h_2=h_2)            
    s_mat = compute_s_matrix(s_mat=s_mat, beta_2=beta_2, variance_mat=var_mat)   
    
    b_max = np.linalg.norm(b_mat, np.inf)
    
    return noisy_gradient_matrix, m_mat, s_mat, b_max



    









        
        
