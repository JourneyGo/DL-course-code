# DL-course-code
Andrew Ng's new course on DL, 2017.  

       第一课
------------------
第一个程序 一个神经元（逻辑回归）logisticRegression.py

按照课程思路实现，比较丑。

第二个程序 一个神经网络（多层感知机）multi_layer_neuro_nets.py

为扩展方便，写成了面向对象形式，支持任意层网络；
17.9.8 增加部分注释；改进测试函数；删除debug信息 


       第二课
------------------
Course 2 Week 1 -- MLP_with_tools.py
（在原有多层感知机程序上扩展，但使用新的文件增加新功能）

完成数据归一化、数据集划分train/dev集合功能、三种正则化方式：L2,Dropout,Earlystoping

17.9.17 增加梯度检查、梯度爆炸/消失预防（完成添加第一周课程提到的全部功能）

一个网络可以如下定义：
```
def buildNet():
    data = loadData()
    #定义网络，选择是否Earlystop，设定全局L2正则化参数lambda;（可选全局化Dropout: Dropout = neuro_keep_ratio）
    #只在Debug时选择 GradCheck = True，非常耗时
    net = NeuroNet(data,Earlystop = True,L2 = 20,GradCheck = True)
    net.addLayer("first",100,0.01)
    net.addLayer("second",80,0.01)
    net.addLayer("third",60,0.01)
    #定义某一层（名称，节点数，学习率，Dropout=neuro_keep_ratio）
    net.addLayer("forth",40,0.02,Dropout = 0.8)
    net.addLayer("fifth",20,0.01)
    net.addLayer("output",1,0.01)
    net.initLayers()
    return net
    
def dothework():
    #创建神经网络
    net = buildNet()
    #迭代训练n次
    net.train(200000)
```
待续……
