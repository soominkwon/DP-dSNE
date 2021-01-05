#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:50:29 2020

@author: soominkwon
"""

import numpy as np
import pylab
from sklearn.utils import shuffle
from adaclip import AdaClip

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X) # this computes the euclidean distance between xi and xj
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def save_gradients(iterations, save_iterations, gradient_matrix):
    """ Saves the gradients from a list of iterations.
    
    """
    
    for i in save_iterations:
        if i == iterations:
            np.savez('gradient_norm_iteration_' + str(i), gradient_matrix)
    
                         
    
def noisy_tsne(X, noise_variance, iters, no_dims=2, perplexity=30.0, 
               PCA=False, initial_dims=50, grad_clip=True):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions.

        Arguments:
            X:              numpy array (data) that is organized by (samples, data)
            noise_variance: Variance of the Gaussian distribution to add noise
            clip_value:     Clips the gradients to have a norm bounded by this value
            no_dims:        Number of dimensions to reduce to
            perplexity:     Perplexity for t-SNE
            PCA:            PCA for pre-processing (should set to False for DP analysis)
            initial_dims:   For PCA... ignore
            grad_clip:      Set True for gradient clipping
        
        Returns:
            Y:              Reduced vector
            dY:             Gradient matrix
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # preprocessing data
    if PCA:
        X = pca(X, initial_dims).real

    # initialize variables
    (n, d) = X.shape
    initial_momentum = 0.3
    final_momentum = 0.9
    eta = 200
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    
    # initialize variables for adaclip
    beta_1 = 0.99
    beta_2 = 0.9
    h_1 = 1e-5
    h_2 = 1e-3
    b_t = []
    m_mat = np.zeros((n, no_dims))
    s_mat = np.ones((n, no_dims)) * np.sqrt(h_1*h_2)

    # computing p-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # running gradient descent iterations
    for iter in range(iters):
        
        # computing pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):            
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            
        # applying gradient clipping
        if grad_clip:
            dY, m_mat, s_mat, b_max = AdaClip(gradient_matrix=dY, noise_variance=noise_variance, m_mat=m_mat, s_mat=s_mat, 
                                       beta_1=beta_1, beta_2=beta_2, h_1=h_1, h_2=h_2)
            
            
        b_t.append(b_max)
        # perform the update with momentum
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
            
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)

        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        if iter == 100:
            P = P / 4.
            
    
    # return solution
    return Y, dY


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    
    # initializing
    var = 1e-3
    iters = 1000

    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    shuf_X, shuf_labels = shuffle(X, labels)
    
    Y, dY, b_maxes = noisy_tsne(X=shuf_X, noise_variance=var, iters=iters, no_dims=2, 
                                perplexity=70.0, PCA=False, initial_dims=50, grad_clip=True)

    pylab.scatter(Y[:, 0], Y[:, 1], 20, shuf_labels)
    pylab.show()
    