### SurvivalModel



#### Problems
- The computation of the negative log-likelihood returns nan due to the integral of the positive duration part
    + It seems decreasing the learning rate fixes it
- Cholesky decomposition of the B matrix used in the construction of the prior covariance sometimes returns error since it turns out a non-positive definite matrix. 
    + To get rid of it, we add a noise term (1e1 * EPS) to the diagonal entries.


### Installation
- pip3 install torch torchvision torchaudio
- pip3 install matplotlib=-3.5
- pip3 install seaborn