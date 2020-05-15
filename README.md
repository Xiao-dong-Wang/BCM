# BCM
Bayesian committee machine

Since Gaussian process regression method requires the inversion of matrices of the dimension of the number of data points which is clearly unsuitable for large data sets, BCM introduces an approximate solution to regression whose computational cost only increases linearly with the number of training patterns. The idea is to split up the data set in M data sets, train M systems on the data sets, and then combine the predictions of the individual systems using a weighting scheme.

Codes reimplemented here is based on the idea from the following paper:

Tresp, Volker. [ACM Press the sixth ACM SIGKDD international conference - Boston, Massachusetts, United States (2000.08.20-2000.08.23)] Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining, - KDD \"00 - The generalized Bayesian committee machine[J]. Neural Computation, 2000.

Dependencies:

GPy: https://github.com/SheffieldML/GPy