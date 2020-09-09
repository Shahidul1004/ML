import numpy as np
def affine_forward(x, w, b):
    out=x.dot(w)+b
    cache=(x, w, b)
    return out, cache

def relu_forward(x):
    out=np.maximum(0, x)
    return out, x

def affine_backward(dout, cache):
    x, w, b=cache
    dw=x.T.dot(dout)
    dx=dout.dot(w.T)
    db=np.sum(dout, axis=0)
    return dx, dw, db

def relu_backward(dout, x):
    dout=(x>0)*dout
    return dout

def batchnorm_forward(x, gamma, beta, mean, variance):
    N=x.shape[0]
    momentum=0.9
    eps=1e-5
    mu=1.0/N*np.sum(x, axis=0)
    xmu=x-mu
    sqxmu=xmu**2
    var=1.0/N*np.sum(sqxmu, axis=0)
    sqrtvar=np.sqrt(var+eps)
    ivar=1.0/sqrtvar
    xhat=xmu*ivar
    xgamma=gamma*xhat
    y=xgamma+beta
    
    mean=momentum*mean+(1-momentum)*mu
    variance=momentum*variance+(1-momentum)*var
    cache=(xhat, gamma, ivar, xmu, sqrtvar, var, eps)
    return y, cache, mean, variance


def batchnorm_backward(dout, cache):
    N, D=dout.shape
    xhat, gamma, ivar, xmu, sqrtvar, var, eps=cache
    
    dbeta=np.sum(dout, axis=0)
    dxgamma=dout
    
    dgamma=np.sum(xhat*dxgamma, axis=0)
    dxhat=gamma*dxgamma
    
    dxmu1=ivar*dxhat
    divar=np.sum(xmu*dxhat, axis=0)
    
    dsqrtvar=-1.0/(sqrtvar*sqrtvar)*divar
    
    dvar=.5*(1.0/np.sqrt(var+eps))*dsqrtvar
    
    dsqxmu=1.0/N*np.ones((N, D))*dvar
    
    dxmu2=2*xmu*dsqxmu
    
    dx1=dxmu1+dxmu2
    
    dmu=-np.sum(dxmu1+dxmu2, axis=0)
    
    dx2=1.0/N*np.ones((N, D))*dmu
    
    dx=dx1+dx2
    
    return dx, dgamma, dbeta

def dropout_forward(x):
    p=.5
    mask=np.random.binomial(1, p, size=x.shape)/p
    out=x*mask
    return out, mask

def dropout_backward(dout, cache):
    mask=cache
    x=dout*mask
    return x