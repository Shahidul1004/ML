import numpy as np
from layers import*
from svm import*
class NeuralNetwork(object):
    
    def __init__(self, num_train, num_feature, hidden_layer, num_class, steps, reg, learning_rate, batchnorm, dropout):
        self.num_train=num_train
        self.layers=np.hstack((num_feature, hidden_layer, num_class))
        self.num_layers=len(self.layers)-1
        self.steps=steps
        self.reg=reg
        self.learning_rate=learning_rate
        self.batchnorm=batchnorm
        self.dropout=dropout
        
        self.params={}
        for i in range(self.num_layers):
            self.params['w%d'%(i+1)]=np.random.randn(self.layers[i], self.layers[i+1])/np.sqrt(self.layers[i]/2)
            self.params['b%d'%(i+1)]=np.zeros(self.layers[i+1])
        
        self.bn_params=[]
        if self.batchnorm:
            #self.bn_params=[{'mode': 'train'} for i in range(self.num_layers - 1)] #not neccesary 
            for i in range(self.num_layers-1):
                self.params['gamma%d'%(i+1)]=np.ones(self.layers[i+1])
                self.params['beta%d'%(i+1)]=np.zeros(self.layers[i+1])
                self.params['mean%d'%(i+1)]=np.zeros(self.layers[i+1])
                self.params['variance%d'%(i+1)]=np.zeros(self.layers[i+1])
                
    def train(self, X_train, Y_train, minibatch, X_test, Y_test):
        
        loss_history=[]
        acc_history=[]
        for step in range(self.steps):
            mask=np.random.choice(self.num_train, minibatch, replace=False)
            scores=X_train[mask]
            
            cache_history=[]
            l2reg=0
            for i in range(self.num_layers):
                w=self.params['w%d'%(i+1)]
                b=self.params['b%d'%(i+1)]
                l2reg+=np.sum(w**2)
            
                scores, cache=affine_forward(scores, w, b)  #wx+b
                cache_history.append(cache)
                
                if i==self.num_layers-1:
                    continue;
                    
                if self.batchnorm:
                    gamma=self.params['gamma%d'%(i+1)]
                    beta=self.params['beta%d'%(i+1)]
                    mean=self.params['mean%d'%(i+1)]
                    variance=self.params['variance%d'%(i+1)]
                    scores, cache, self.params['mean%d'%(i+1)], self.params['variance%d'%(i+1)]=batchnorm_forward(scores, gamma, beta, mean, variance)
                    cache_history.append(cache)
                scores, cache=relu_forward(scores)  #max(0, scores)
                cache_history.append(cache)
                if self.dropout:
                    scores, cache=dropout_forward(scores)
                    cache_history.append(cache)

                    
            l2reg*=.5*self.reg
            loss, dout=svm_loss(scores, Y_train[mask])
            loss+=l2reg
            grads={}
            
            while i>=0:
                if i!=self.num_layers-1:
                    if self.dropout:
                        dout=dropout_backward(dout, cache_history.pop())
                    dout=relu_backward(dout, cache_history.pop())
                    if self.batchnorm:
                        dout, grads['gamma%d'%(i+1)], grads['beta%d'%(i+1)]=batchnorm_backward(dout, cache_history.pop())
                dout, grads['w%d' %(i+1)], grads['b%d'%(i+1)]=affine_backward(dout, cache_history.pop())
                grads['w%d'%(i+1)]+=2*self.reg*self.params['w%d'%(i+1)]
            
                i-=1
        
            for i in range(self.num_layers):
                self.params['w%d'%(i+1)]-=self.learning_rate*grads['w%d'%(i+1)]
                self.params['b%d'%(i+1)]-=self.learning_rate*grads['b%d'%(i+1)]
                if i==self.num_layers-1:
                    continue;
                if self.batchnorm:
                    self.params['gamma%d'%(i+1)]-=self.learning_rate*grads['gamma%d'%(i+1)]
                    self.params['beta%d'%(i+1)]-=self.learning_rate*grads['beta%d'%(i+1)]
                    
            loss_history.append(loss)
            temp=self.predict(X_test, Y_test)/Y_test.shape[0]
            acc_history.append(temp)
            print(step, loss, temp)
            
        return loss_history, grads, acc_history
    
    def predict(self, X_test, Y_test):
        scores=X_test
        for i in range(self.num_layers):
            scores=scores.dot(self.params['w%d'%(i+1)])+self.params['b%d'%(i+1)] #affine_forward
            if i==self.num_layers-1:
                continue;
            if self.batchnorm:
                xhat=(scores-self.params['mean%d'%(i+1)])/np.sqrt(self.params['variance%d'%(i+1)]+1e-5)
                scores=xhat*self.params['gamma%d'%(i+1)]+self.params['beta%d'%(i+1)]
            scores=np.maximum(scores, 0)
                
        mx=np.argmax(scores, axis=1)
        #print(mx, Y_test)
        acc=np.sum(mx==Y_test)
        return acc