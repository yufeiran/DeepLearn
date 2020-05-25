import numpy as np
import sys,os
sys.path.append(os.pardir)
sys.path.append("..")
from dataset.mnist import load_mnist


def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta=1e-7
    return -np.sum(t*np.log(y+delta))

(x_train,t_train),(x_test,t_test)=\
    load_mnist(normalize=True,one_hot_label=True)

print(x_train.shape) #(60000,784)
print(t_train.shape) #(60000,10)