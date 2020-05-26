import numpy as np
import matplotlib.pylab as plt

import sys,os
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist


def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    batch_size=y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size
    
def numerical_diff(f,x):
    h=1e-4 #0.0001
    return (f(x+h)-f(x-h))/(2*h)

def numerical_gradient(f,x):
    h=1e-4 #0.0001
    grad=np.zeros_like(x) #生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val=x[idx]
        # f(x+h)的计算
        x[idx]=tmp_val+h
        fxh1=f(x)

        # f(x-h)的计算
        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val #还原值
    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x 

    for i in range(step_num):
        grad=numerical_gradient(f,x)
        x-=lr*grad 
    return x

def function_1(x):
    return 0.01*x**2+0.1*x

def function_k(x,x0,y0,k):
    return k*(x-x0)+y0

def function_2(x):
    return x[0]**2+x[1]**2

def function_temp1(x0):
    return x0*x0+4.0**2.0

def function_temp2(x1):
    return 3.0**2.0+x1*x1

(x_train,t_train),(x_test,t_test)=\
    load_mnist(normalize=True,one_hot_label=True)

print(x_train.shape) #(60000,784)
print(t_train.shape) #(60000,10)

train_size=x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)
x_batch=x_train[batch_mask]
y_batch=t_train[batch_mask]

x=np.arange(0.0,20.0,0.1) #以0.1为单位，从0到20的数组x
y=function_1(x)
y1=function_k(x,5,function_1(5), numerical_diff(function_1,5))
y2=function_k(x,10,function_1(10),numerical_diff(function_1,10))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()
print(numerical_diff(function_1,5))
print(numerical_diff(function_1,10))
print(numerical_diff(function_temp1,3.0))
print(numerical_diff(function_temp2,4.0))
print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([0.0,2.0])))
print(numerical_gradient(function_2,np.array([3.0,0.0])))

init_x=np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))