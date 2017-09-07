# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 19:02:19 2017
Following the course of Andrew.Ng, DeepLearning I
Second step a multi-layer neuron nets
@author: Gao
"""
import numpy as np
import pandas as pd 
class NeuroLayer:
    def __init__(self,name,num_units,alpha):
        self.name = name
        self.num_units = num_units
        self.alpha = alpha
        self.weights = "nan"
        self.inputs = "nan"
        self.act = "nan"
        self.derivative_Z = "nan"
        self.delta = "nan"
        
    def activate(self,Z):
        A = 1.0/(1+np.exp(-Z))
        return A
        
    def computeA(self,input_x):
        assert(self.weights.shape[1] == input_x.shape[0])
        self.inputs = input_x
        z=np.dot(self.weights,input_x)
        self.act = self.activate(z)
        
    def computeDZ(self,input_delta):
        derivative_A_on_Z = self.act*(1-self.act) 
        assert(input_delta.shape == derivative_A_on_Z.shape)
        self.derivative_Z = input_delta*derivative_A_on_Z
        
    def computeLoss(self,input_y):
        self.derivative_Z = self.act - input_y.T
        
    def computeDelta(self):
        assert(self.weights.shape[0] == self.derivative_Z.shape[0])
        self.delta = np.dot(self.weights.T,self.derivative_Z)
        self.delta = np.delete(self.delta, (-1), axis=0)
        
    def updataWeights(self):
        DW = np.dot(self.derivative_Z,self.inputs.T)*1.0/self.inputs.shape[1]
        assert(DW.shape == self.weights.shape)
        self.weights = self.weights - self.alpha * DW
        
    def initParams(self,last_layer_units):
        self.weights = np.random.randn(self.num_units,last_layer_units+1)
        
    
class NeuroNet:
    def __init__(self, train_set, train_label, test_set, test_label):
        #train/test set is arranged by (num_of_feature, num_of_dataset)
        #train/test label is vertical vector, namey (num_of_dataset, 1)
        self.train_set = train_set
        self.train_label = train_label
        self.test_set = test_set
        self.test_label = test_label
        #useful while initialize the layers
        self.num_of_dataset = train_set.shape[1]
        self.num_of_feature = train_set.shape[0]
        self.layers = []
    def addLayer(self,name,num_units,alpha):
        layer = NeuroLayer(name,num_units,alpha)
        self.layers.append(layer)
        
    def initLayers(self):
        last_layer_units = self.num_of_feature
        for layer in self.layers:
            layer.initParams(last_layer_units)
            last_layer_units = layer.num_units
        
    def forward(self,input_x):
        feed_data = input_x
        for layer in self.layers:
            #add bias through augmentation of the Weight-matrix
            new_row = np.ones((1,feed_data.shape[1]))
            feed_data = np.row_stack((feed_data,new_row))
            layer.computeA(feed_data)
            feed_data = layer.act
    
    def backwark(self,input_y):
        layers = self.layers[::-1]
        delta = "nan"
        for layer in layers:
            if layer.name == "output":
                layer.computeLoss(input_y)
            else:
                layer.computeDZ(delta)
            layer.computeDelta()
            delta = layer.delta
            layer.updataWeights()
            
    def train(self,epoch_limits):
        print "training start..."
        for i in xrange(epoch_limits):
            self.forward(self.train_set)
            self.backwark(self.train_label)
            print("training %05d epoch: \n\tprecision on train %02.02f \n\tprecision on test  %02.02f"%(i+1,self.stats("train"),self.stats("test")))

            
    def stats(self,type_of_dataset):
        if type_of_dataset == "train":
            data = self.train_set
            label = self.train_label
        else:
            data = self.test_set
            label = self.test_label
        self.forward(data)
        layer = self.layers[-1]
        y = layer.act
        y[y >  0.5] = 1
        y[y <= 0.5] = 0
        return sum(y.T==label)*1.0/label.shape[0]
        
    def predict(self):
        self.forward(self.test_set)
        layer = self.layers[-1]
        y = layer.act
        y[y >  0.5] = 1
        y[y <= 0.5] = 0
        return sum(y.T==self.test_label)*1.0/self.test_label.shape[0],y            

def loadData():
    '''
    读取数据
    -- X为训练集，组织形式：每行一个样本，列个数即特征维度
    -- y为训练集标签，与训练集逐行对应
    -- Tx,Ty为测试集，同理
    '''
    train_set = pd.read_csv("X.csv",header=None)
    train_label = pd.read_csv("y.csv",header=None)
    test_set = pd.read_csv("Tx.csv",header=None)
    test_label = pd.read_csv("Ty.csv",header=None)
    return train_set,train_label,test_set,test_label       
        
def buildNet():
    Train,Ty,test,ty = loadData()
    #根据课程内容，将数据集转置，调整成每列一个样本
    #标签集保持列向量形式
    net = NeuroNet(Train.values.T,Ty.values,test.values.T,ty.values)
    #层参数：名称，单元个数，学习率
    net.addLayer("first",20,0.01)
    net.addLayer("second",10,0.01)
    #最后一层请务必叫“output”
    net.addLayer("output",1,0.01)
    net.initLayers()
    return net
    
def dothework():
    #创建神经网络
    net = buildNet()
    print("build success,net have %d layers"%len(net.layers))
    #迭代训练 5000次
    net.train(5000)

if __name__ == "__main__":
    dothework()