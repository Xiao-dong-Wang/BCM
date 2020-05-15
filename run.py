import autograd.numpy as np
import sys
import toml
from get_dataset import *
import GPy

argv = sys.argv[1:]
conf = toml.load(argv[0])

### Load data
name = conf['funct']
funct = get_funct(name)
num = conf['num']
bounds = np.array(conf['bounds'])
dim = bounds.shape[0]
bfgs_iter = conf['bfgs_iter']

### GP training
dataset1, samples = init_dataset(funct, num, bounds)
#print('dataset1',dataset1)
#print('samples',samples)
train_x1 = dataset1['train_x'].T
train_y1 = dataset1['train_y'].T

k = GPy.kern.RBF(dim)
model1 = GPy.models.GPRegression(X=train_x1, Y=train_y1,kernel=k)
model1.kern.variance = np.var(train_y1)
model1.kern.lengthscale = np.std(train_x1)
model1.likelihood.variance = 0.01 * np.var(train_y1)
model1.optimize()

### Test 
nn = 200
testdata = get_test(funct, nn, bounds)
X_star = testdata['test_x']
y_star = testdata['test_y']
y_pred1, y_var1 = model1.predict(X_star.T)
#print('y_star',y_star)
#print('y_pred1',y_pred1)

### BCM
samples_x = samples['samples_x']
samples_y = samples['samples_y']
n_clus = samples_x.shape[0]
sum_inv = np.zeros((nn,1))
sum_rat = np.zeros((nn,1))
for i in range(n_clus):
    train_x2 = samples_x[i].T
    train_y2 = samples_y[i].reshape(-1,1)

    k = GPy.kern.RBF(dim)
    model = GPy.models.GPRegression(X=train_x2, Y=train_y2,kernel=k)
    model.kern.variance = np.var(train_y2)
    model.kern.lengthscale = np.std(train_x2)
    model.likelihood.variance = 0.01 * np.var(train_y2)
    model.optimize()

    y_pred, y_var = model.predict(X_star.T) 
    sum_inv = sum_inv + 1.0/y_var
    sum_rat = sum_rat + y_pred/y_var

y_var2 = 1.0/sum_inv
y_pred2 = y_var2 * sum_rat
#print('y_pred2',y_pred2)

### Accuracy
rmse1 = np.linalg.norm(y_pred1.T - y_star)
rmse2 = np.linalg.norm(y_pred2.T - y_star)
print('rmse1',rmse1)
print('rmse2',rmse2)
    








