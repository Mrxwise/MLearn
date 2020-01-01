import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Two-dimension linear regression
def TwoDimension(dataSet):          # 基于公式直接求解 ω 与 b：基于均方差损失函数E(ω,b) = ∑(yi - ωxi - b)^2 
    average_x = sum([data[0] for data in dataSet]) / len(dataSet)   #E(ω,b)分别求ω,b的偏导并令为0，化解后得到相关 ω,b 的计算公式
    omiga = sum([data[1] * (data[0] - average_x) for data in dataSet]) / (sum([data[0] * data[0] for data in dataSet]) - sum([data[0] for data in dataSet])** 2 / len(dataSet))
    b = sum([data[1] - omiga * data[0] for data in dataSet]) / len(dataSet)
    return omiga, b
# Code view
# data_set = [[1,3],[4,6],[2,5],[7,10]]
# omiga, b = TwoDimension(data_set)
# for point in data_set:
#     plt.scatter(point[0], point[1])
# x = np.arange(-10, 10, 0.1)
# y = omiga * x + b
# plt.plot(x,y)
# plt.show()


# Muti-Dimension linear regression

def MutiDimension(dataSet):     # 多元线性回归
    datay = dataSet[:, -1]  #取得所有数据的labels
    datax = dataSet[:, 0:-1]  #取所有行的除了最后一行数据，即X    
    dataX = np.insert(datax,len(datax[0]),[1 for i in range(len(datax))],1)             #需要将X* = (X,1) 以在后面的矩阵乘法中引入b的位置
    omiga_star = np.dot(np.dot(np.linalg.inv(np.dot(dataX.T,dataX)),dataX.T),datay)     #由公式得到ω* = (ω, b):同样基于多元情况下的均方差 E = argmin (y - Xω*)^T(y - Xω*)
                                                                                        #其中y,ω为向量，X为矩阵,b为常量，令E对ω*求导为0，当E为极小值时ω*的最优解
    return omiga_star                                                                   #回归模型令x* = (x,1), f(x*) = x* × ω* = x*(X.T * X)^-1 * X.T * y
                                                                     
# Code View
#data_set = np.array([[1,2,3,4,35],[2,3,4,5,45],[5,2,3,2,31],[12,42,53,2,268],[12,55,6,34,281],[0,1,2,3,25]])
#print(MutiDimension(data_set))

#Linear classification -> logit regression
def Sigmoid(x):
    return 1 / (1 + math.exp(-1 * x))
def LogitLossGradient(dataSet, B):              # 对Logit损失函数的导函数
    datax = dataSet[:, 0:-1]
    datay = dataSet[:, -1]
    dataX = np.insert(datax, len(datax[0]), [1 for i in range(len(datax))], 1)
    ret = []
    for i in range(len(dataSet)):
        if (np.dot(B.T, dataX[i]) < 100):       #若math.exp(n)的值过大，则会溢出
            first = math.exp(np.dot(B.T, dataX[i]))
            second = first / (1 + first)
        else:second = 1
        ret.append(np.dot(dataX[i], (datay[i] - second)))
    ans = sum(ret)                          # 返回导数值
    return ans
def LogitGradientDown(dataSet, B, Step):      # 梯度下降求解最优参数B，Step为梯度下降步长(学习率)
    Next_B = B + LogitLossGradient(dataSet, B) * Step    # 递归求解最优B
    if (np.linalg.norm(Next_B - B) < 0.01):  #计算B 与 Next_B的相似度
        return Next_B
    else:
        return LogitGradientDown(dataSet,Next_B,Step)
def LogitRegression(dataSet,X):         #逻辑回归，其中dataSet必须是np.array类型的，X可以为数组
    B = np.random.rand(len(dataSet[0]))
    B = LogitGradientDown(dataSet, B, 0.1)  
    ret = []
    for x in X:
        x.append(1)
        if (Sigmoid(np.dot(B.T, x)) > 0.5):
            ret.append(1)
        else: ret.append(0)
    return ret

# Code View
# from sklearn.linear_model import LogisticRegression   #和sklearn中的逻辑回归模型进行比较
# import random
# data_set = [[random.random() * 10 , random.random() * 10 , 1] for i in range(20)]
# data_set += [[random.random() * 10 - 5, random.random() * 10 - 5, 0] for i in range(10)]
# data_set = np.array(data_set)
# test_set = [[random.random(), random.random()] for z in range(5)]

# model = LogisticRegression()
# model.fit(data_set[:, 0:-1], data_set[:, -1])
# print(model.predict(test_set))
# print(LogitRegression(data_set,test_set))