import autograd.numpy as np
import os
import string

### Generate binary numbers to describe boundaries
def genbinnum(dim):
    ret = np.zeros((2**dim, dim))
    for i in range(2**dim):
        tmp = np.zeros(dim)
        num = i
        j = dim-1
        while num != 0:
            a = num % 2
            num = int(num/2)
            tmp[j] = a
            j = j-1
            if j < 0:
                break
        ret[i,:] = tmp
    return ret

def init_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    Binset = genbinnum(dim)
    Boundset = np.zeros((2**dim, dim, 2))
    clus_num = int(num/2**dim)
    samples_x = np.zeros((2**dim, dim, clus_num))
    ### Divide the space into 2**dim parts
    for i in range(2**dim):
        for j in range(dim):
            if Binset[i,j] == 0:
                Boundset[i,j] = [-0.5, 0]
                samples_x[i,j,:] = np.random.uniform(-0.5, 0, clus_num)
            else:
                Boundset[i,j] = [0, 0.5]
                samples_x[i,j,:] = np.random.uniform(0, 0.5, clus_num)

    samples_y = np.zeros((2**dim, clus_num))
    for i in range(2**dim):
        samples_y[i] = funct(samples_x[i], bounds)
    
    x = samples_x[0]
    y = samples_y[0]
    for i in range(2**dim-1):
        x = np.hstack((x, samples_x[i+1]))
        y = np.hstack((y, samples_y[i+1]))
    dataset = {}
    dataset['train_x'] = x
    dataset['train_y'] = y.reshape(1,-1)
    samples = {}
    samples['samples_x'] = samples_x
    samples['samples_y'] = samples_y
    return dataset, samples

def get_test(funct, num, bounds):
    dim = bounds.shape[0]
    dataset = {}
    dataset['test_x'] = np.random.uniform(-0.5, 0.5, (dim, num))
    dataset['test_y'] = funct(dataset['test_x'], bounds)
    return dataset

### Three test cases
## bound:[0,1]
def sin1(x, bounds):
    tmp = sin1_src(x, bounds)
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = (x-np.sqrt(2))*tmp**2
    return ret.reshape(1, -1)

def sin1_src(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = np.sin(8.0*np.pi*x)
    return ret.reshape(1, -1)

## bound:[0,1]
def sin2(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = (6.0*x - 2.0)**2 * np.sin(12.*x - 4.0)
    return ret.reshape(1, -1)

## bound:[[-5,10],[0,15]]
def branin(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    tmp1 = -1.275*np.square(x[0]/np.pi) + 5*x[0]/np.pi + x[1] - 6
    tmp2 = (10 - 5/(4*np.pi))*np.cos(x[0])
    ret = tmp1*tmp1 + tmp2 + 10
    return ret.reshape(1,-1)

def get_funct(funct):
    if funct == 'branin':
        return branin
    elif funct == 'sin1':
        return sin1
    elif funct == 'sin2':
        return sin2
    else:
        return sin1

