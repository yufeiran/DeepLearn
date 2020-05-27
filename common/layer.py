import numpy as np
import sys,os 
sys.path.append(os.getcwd())
from common.functions import *
from common.gradient import *
class MulLayer:
    def __init__(self):
        self.x=None 
        self.y=None 
    
    def forward(self,x,y):
        self.x=x 
        self.y=y 
        out=x*y 

        return out 
    
    def backward(self,dout):
        dx=dout*self.y #翻转x,y
        dy=dout*self.x 
        return dx,dy 

class AddLayer:
    def __init__(self):
        pass 

    def forward(self,x,y):
        out=x+y 
        return out 
    
    def backward(self,dout):
        dx=dout*1 
        dy=dout*1
        return dx,dy 

class Relu:
    def __init__(self):
        self.mask=None 
    def forward(self,x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        return out 
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout 
        return dx

class Sigmoid:
    def __init__(self):
        self.out=None 
    
    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out 
        return out
    
    def backward(self,dout):
        dx=dout*(1.0-self.out)*self.out 
        return dx

class Affine:
    def __init__(self,W,b):
        self.W=W
        self.b=b 
        self.x=None 
        self.original_x_shape=None 
        self.dW=None 
        self.db=None 
    def forward(self,x):
        self.original_x_shape=x.shape 
        
        x=x.reshape(x.shape[0],-1)
        self.x=x
        out=np.dot(x,self.W)+self.b 
        return out 
    def backward(self,dout):
        dx=np.dot(dout,self.W.T)
        self.dW=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)

        dx=dx.reshape(*self.original_x_shape) #还原输入数据的形状
        return dx 
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None #损失 
        self.y=None #softmax输出
        self.t=None #监督数据(one-hot vector)
    
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss =cross_entropy_error(self.y,self.t) 
        return self.loss 
    
    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        if self.t.size==self.y.size: #监督数据是one-hot-vector的情况
            dx=(self.y-self.t)/batch_size
        else:
            dx=self.y.copy() 
            dx[np.arange(batch_size),self.t]-=1
            dx=dx/batch_size
        return dx 