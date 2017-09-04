# -*- coding: utf-8 -*-
"""
Created on Sun Sep 03 12:25:30 2017
Following the course of Andrew.Ng, DeepLearning I
First step a single neuron -- logistic regression
@author: Gao
"""
import numpy as np
import pandas as pd 
    
   
def activate(Z):
    A = 1.0/(1+np.exp(-Z))
    return A
    
def train(epoch_limits, alpha, W, train_set, y):
    J = []
    for iter in xrange(epoch_limits):
        assert(W.shape[1]==train_set.shape[0])
        Z = np.dot(W,train_set)
        A = activate(Z)
        
        a_first = np.log(A)
        a_second = np.log(1-A)
        L_first= np.dot(a_first,y)
        L_second = np.dot(a_second,1-y)
        L = (L_first+L_second)*-1
        J.append(L[0][0])
        
        dZ = A - y.T
        dW = np.dot(dZ,train_set.T)*1.0/train_set.shape[1]
        W = W - alpha*(1.0-(1+iter)/epoch_limits)* dW
        print("epoch: %05d cost %.02f"%(iter,L[0][0]))
        precision = test(train_set,y,W)
        print("precision on train set: %.02f"%precision)
    return W
    
def predict(X,W):
    assert(X.shape[0]==W.shape[1])
    y = activate(np.dot(W,X))
    y[y>0.5] = 1
    y[y<=0.5] = 0
    return y
    
def loadData():
    train_set = pd.read_csv("X.csv",header=None)
    train_label = pd.read_csv("y.csv",header=None)
    test_set = pd.read_csv("Tx.csv",header=None)
    test_label = pd.read_csv("Ty.csv",header=None)
    return train_set,train_label,test_set,test_label
    
def test(val_set,val_label,params):
    result = predict(val_set,params)
    return sum(result.T==val_label)*1.0/val_set.shape[1]
    
def do():
    train_set,train_label,test_set,test_label = loadData()
    #add a bias columns
    train_set.loc[:,'bias']=pd.Series(np.ones(train_set.shape[0]))
    test_set.loc[:,'bias']=pd.Series(np.ones(test_set.shape[0]))
    #initial params
    W = np.random.randn(1,train_set.shape[1])
    alpha = 0.01
    epoch = 40000
    W = train(epoch,alpha,W,train_set.values.T,train_label.values)
    print("Test precision: %.2f"%test(test_set.values.T,test_label.values,W))
    
if __name__=='__main__':
    do()