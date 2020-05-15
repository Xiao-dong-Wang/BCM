import autograd.numpy as np
import sys
import toml
from get_dataset import *
import pickle
import matplotlib.pyplot as plt
import GPy

np.random.seed(1234)

argv = sys.argv[1:]
conf = toml.load(argv[0])

name = conf['funct']
funct = get_funct(name)
num = conf['num']
bounds = np.array(conf['bounds'])
dim = bounds.shape[0]
bfgs_iter = conf['bfgs_iter']

### GP
dataset1, samples = init_dataset(funct, num, bounds)
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
X_star = np.linspace(-0.5, 0.5, nn)[None,:]
y_star = funct(X_star, bounds)
y_pred1, y_var1 = model1.predict(X_star.T)

### BCM
samples_x = samples['samples_x']
samples_y = samples['samples_y']

train_x21 = samples_x[0].T
train_y21 = samples_y[0].reshape(-1,1)
k = GPy.kern.RBF(dim)
model = GPy.models.GPRegression(X=train_x21, Y=train_y21,kernel=k)
model.kern.variance = np.var(train_y21)
model.kern.lengthscale = np.std(train_x21)
model.likelihood.variance = 0.01 * np.var(train_y21)
model.optimize()
y_pred21, y_var21 = model.predict(X_star.T) 

train_x22 = samples_x[1].T
train_y22 = samples_y[1].reshape(-1,1)
k = GPy.kern.RBF(dim)
model = GPy.models.GPRegression(X=train_x22, Y=train_y22,kernel=k)
model.kern.variance = np.var(train_y22)
model.kern.lengthscale = np.std(train_x22)
model.likelihood.variance = 0.01 * np.var(train_y22)
model.optimize()
y_pred22, y_var22 = model.predict(X_star.T) 

y_var2 = 1.0/(1.0/y_var21 + 1.0/y_var22)
y_pred2 = y_var2 * (y_pred21/y_var21 + y_pred22/y_var22)

train_x1_real = train_x1 * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
train_x21_real = train_x21 * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
train_x22_real = train_x22 * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
X_star_real = X_star * (bounds[0,1]-bounds[0,0]) + (bounds[0,1]+bounds[0,0])/2
print(train_x1_real,train_x21_real,train_x22_real,X_star_real)
print(train_y1,train_y21,train_y22,y_star)

plt.figure(1)
plt.cla()
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.plot(X_star_real.flatten(), y_star.flatten(), 'b-', label = "latent function", linewidth=2)
plt.plot(X_star_real.flatten(), y_pred1.flatten(), 'r--', label = "GP Prediction", linewidth=2)
lower1 = y_pred1 - 2.0*np.sqrt(y_var1)
upper1 = y_pred1 + 2.0*np.sqrt(y_var1)
plt.fill_between(X_star_real.flatten(), lower1.flatten(), upper1.flatten(), 
                 facecolor='pink', alpha=0.5, label="Two std band")
plt.plot(train_x1_real, train_y1, 'ko')
plt.legend()
ax = plt.gca()
ax.set_xlim([bounds[0,0], bounds[0,1]])
plt.xlabel('x')
plt.ylabel('f(x)')

plt.figure(2)
plt.cla()
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.plot(X_star_real.flatten(), y_star.flatten(), 'b-', label = "latent function", linewidth=2)
plt.plot(X_star_real.flatten(), y_pred2.flatten(), 'r--', label = "BCM Prediction", linewidth=2)
lower2 = y_pred2 - 2.0*np.sqrt(y_var2)
upper2 = y_pred2 + 2.0*np.sqrt(y_var2)
plt.fill_between(X_star_real.flatten(), lower2.flatten(), upper2.flatten(), 
                 facecolor='pink', alpha=0.5, label="Two std band")
plt.plot(train_x21_real, train_y21, 'ko')
plt.plot(train_x22_real, train_y22, 'ko')
plt.legend()
ax = plt.gca()
ax.set_xlim([bounds[0,0], bounds[0,1]])
plt.xlabel('x')
plt.ylabel('f(x)')

plt.figure(3)
plt.cla()
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.plot(X_star_real.flatten(), y_star.flatten(), 'b-', label = "latent function", linewidth=2)
plt.plot(X_star_real.flatten(), y_pred21.flatten(), 'r--', label = "part1 Prediction", linewidth=2)
lower21 = y_pred21 - 2.0*np.sqrt(y_var21)
upper21 = y_pred21 + 2.0*np.sqrt(y_var21)
plt.fill_between(X_star_real.flatten(), lower21.flatten(), upper21.flatten(), 
                 facecolor='pink', alpha=0.5, label="Two std band")
plt.plot(train_x21_real, train_y21, 'ko')
plt.legend()
ax = plt.gca()
ax.set_xlim([bounds[0,0], bounds[0,1]])
plt.xlabel('x')
plt.ylabel('f(x)')

plt.figure(4)
plt.cla()
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)
plt.plot(X_star_real.flatten(), y_star.flatten(), 'b-', label = "latent function", linewidth=2)
plt.plot(X_star_real.flatten(), y_pred22.flatten(), 'r--', label = "part2 Prediction", linewidth=2)
lower22 = y_pred22 - 2.0*np.sqrt(y_var22)
upper22 = y_pred22 + 2.0*np.sqrt(y_var22)
plt.fill_between(X_star_real.flatten(), lower22.flatten(), upper22.flatten(), 
                 facecolor='pink', alpha=0.5, label="Two std band")
plt.plot(train_x22_real, train_y22, 'ko')
plt.legend()
ax = plt.gca()
ax.set_xlim([bounds[0,0], bounds[0,1]])
plt.xlabel('x')
plt.ylabel('f(x)')

plt.show()






