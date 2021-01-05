# DP-dSNE

This repository contains scripts for decentralized analysis of neuroimaging datasets similar to paper: https://www.biorxiv.org/content/10.1101/826974v2.abstract

This new paper is currently under review for submission. This paper ensures formal privacy guarantees where the data at each local site is private thus precluding
centralized analyses. The privacy guarantees are made with using the recently proposed AdaCliP algorithm.

AdaCliP Paper: https://arxiv.org/abs/1908.07643

## Programs
The following is a brief explanation for each script:

* adaclip_tsne.py - Centralized t-SNE with formal privacy guarantees
* adaclip.py - Implementation of AdaCliP for private Stochastic Gradient Descent (SGD)
* generate_synth_data.py - Generates synthetic data for private centralized t-SNE
* example.py - Example of centralized t-SNE

## Tutorial
This tutorial can be found in example.py:

```
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
```

