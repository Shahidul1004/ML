
import numpy as np

def svm_loss(scores, Y_train):

    num_train=Y_train.shape[0]
    correct_scores=scores[range(num_train), Y_train]
    correct_scores=correct_scores.reshape(correct_scores.shape[0], -1)
    scores=scores-correct_scores+1
    scores=np.fmax(scores, 0)
    scores[range(num_train), Y_train]=0
    loss=np.sum(scores)
    #print(loss)
    loss/=num_train
    
    temp=np.sum(scores>0, axis=1)
    dx=np.zeros_like(scores)
    dx[scores>0]=1;
    dx[range(num_train), Y_train]=-temp
    dx/=num_train
    
    return loss, dx