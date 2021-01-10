# BCM

## About
Bayesian committee machine

Since Gaussian process regression method requires the inversion of covariance matrix which is unsuitable for large data sets, BCM introduces an approximate solution to regression whose computational cost only increases linearly with the number of training patterns. The basic idea is to partition the data set into M data sets, train M systems on the data sets, and then combine the predictions of the individual systems using a weighting scheme.

The regression results comparisons between conventional Gaussian process method and BCM method are shown here.

![image](https://github.com/Xiao-dong-Wang/BCM/blob/master/figures/GP.png)

![image](https://github.com/Xiao-dong-Wang/BCM/blob/master/figures/BCM.png)

Codes reimplemented here is based on the idea from the following paper:

- V. Tresp. A Bayesian committee machine. *Neural Computation*, 12:2719–2741, 2000.

## Dependencies:

GPy: https://github.com/SheffieldML/GPy