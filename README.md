### Unknown Model Name
This is a PyTorch implementation of the paper "Unknown Model Name" by Unknown Authors.

### Installation
- Initialize a new conda environment and activate it.
```
conda create -n survival python=3.11
conda activate survival
```
- You can install all the required packages by running the following command.
```
pip install -r requirements.txt
```

**Note** : Please visit the [PyTorch website](https://pytorch.org/get-started/locally/) for the installation details of the PyTorch library (version Stable 2.0.0 with possible CUDA 11.7 support) regarding your system.


#### Problems
- The computation of the negative log-likelihood returns nan due to the integral of the positive duration part
    + It seems decreasing the learning rate fixes it
- Cholesky decomposition of the B matrix used in the construction of the prior covariance sometimes returns error since it turns out a non-positive definite matrix. 
    + To get rid of it, we add a noise term (1e1 * EPS) to the diagonal entries.

