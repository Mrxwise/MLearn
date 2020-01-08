import numpy as np
import pandas as pd
import random
import copy
import math

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dSigmoid(x):
    return Sigmoid(x)*(1-Sigmoid(x))
# Multi-layer Perceptron
def MLP(inputX, outputY, HiddenLayer):      #初始化MLP网络
    '''
        inputX->int  给定MLP网络输入节点数
        outputY->int 给定MLP网络输出节点数
        HiddenLayer->List 给定隐藏层节点数
    '''
    weight_matrix_layer = []    # 每一个矩阵代表每一层的weight
    bias_vector_layer = []  # 每一个向量代表每一层的bias
    
    #初始化随机参数
    if (len(HiddenLayer) == 0):
        weight_matrix_layer.append(np.random.rand(inputX, outputY))
        bias_vector_layer.append(np.random.rand(outputY))
    else:
        for i in range(len(HiddenLayer) + 1):
            if (i == 0):
                weight_matrix_layer.append(np.random.rand(inputX, HiddenLayer[i]))
                bias_vector_layer.append(np.random.rand(HiddenLayer[i]))
            elif (i == len(HiddenLayer)):
                weight_matrix_layer.append(np.random.rand(HiddenLayer[i-1], outputY))
                bias_vector_layer.append(np.random.rand(outputY))
            else:
                weight_matrix_layer.append(np.random.rand(HiddenLayer[i - 1], HiddenLayer[i]))
                bias_vector_layer.append(np.random.rand(HiddenLayer[i]))
    #输出随机化模型
    return weight_matrix_layer, bias_vector_layer
def modelUpdate(weight, bias, dw, db, learn_rate):
    for index in range(len(weight)):
        for x in range(weight[index].shape[0]):
            for y in range(weight[index].shape[1]):
                weight[index][x][y] -= learn_rate * dw[index][x][y]
        for z in range(bias[index].shape[0]):
            if (len(bias[index].shape) == 1):
                bias[index][z] -= learn_rate*db[index][z]
            else:
                h = bias[index][1]
                for h in range(h):
                    bias[index][z][h] -= learn_rate*db[index][z][h]
    return weight,bias
def backword(weight_layers, bias_layers, trainX, trainY, learn_rate):
    '''
        weight_layers->List 权重矩阵
        bias_layers->List 偏置矩阵
        trainX->numpy.arrays 数据X
        trainY->numpy.arrays 数据Y
        learn_rate->float 学习率
        来源 http://andyjin.applinzi.com/?p=397
    '''
    if (len(weight_layers) != len(bias_layers)): raise Exception("Layers Error!")
    layer_n = len(weight_layers)
    Loss = 0
    #F = random.randint(0,len(trainX)-1)
    #for i in range(F,F+1):
    for i in range(4):#len(trainX)):
        yk = trainX[i]
        save_zi = [list(trainX[i])]  #记录zi
        save_ai = [list(trainX[i])]
        #save_ai = [list(map(Sigmoid, trainX[i]))]   #记录ai
        for l in range(layer_n):
            yk = np.array(np.dot(yk, weight_layers[l]) + bias_layers[l])  # 得到每一层的yk为一个向量，其中的值代表每一层节点的值
            save_zi.append(yk)
            yk = np.array(list(map(Sigmoid, yk)))   # 将对应的值取Sigmoid
            save_ai.append(list(yk))  # 前向传播计算每一层的yi值
        Loss += 1/2*np.linalg.norm(trainY[i]-yk)
        dlossL = (yk - trainY[i]) * np.array(list(map(dSigmoid, save_zi[-1])))  # 由δL = (aL - y) * σ΄'(zL) 计算L层的损失 δL = dC / dzL
        dlossMatrix = [np.array(dlossL)]
        index = 0
        for z in range(layer_n - 1, 0, -1):
            dlossI = np.dot(dlossMatrix[index],weight_layers[z].T) * np.array(list(map(dSigmoid, save_zi[z])))
            index += 1
            dlossMatrix.append(dlossI)
        dlossMatrix.reverse()

        dbiasMatrix = dlossMatrix
        dweightMatrix = []
        count = 0

        for M in dlossMatrix:
            dweightMatrix.append(np.zeros([len(save_ai[count]), len(M)]).astype(np.float64))
            for i in range(dweightMatrix[-1].shape[0]):
                for j in range(dweightMatrix[-1].shape[1]):
                    dweightMatrix[-1][i][j] = M[j] * save_ai[count][i]
            count += 1
        weight_layers,bias_layers = modelUpdate(weight_layers, bias_layers, dweightMatrix, dbiasMatrix, learn_rate)
    return Loss/len(trainX)
def train(weight_layers, bias_layers, trainX, trainY, Step, learn_rate):
    loss_list = []
    for step in range(Step):
        loss = backword(weight_layers, bias_layers, trainX, trainY, learn_rate)
        if(step % 10 == 0):loss_list.append((step,loss))
        if (step % 100 == 0):
            print("learn steps %d, Loss %f" % (step,loss))
    return weight_layers,bias_layers,loss_list
def predict(w, b, testX):
    layer_n = len(w)
    ret = []
    for data in testX:
        yk = data
        for l in range(layer_n):
            yk = np.dot(yk, w[l]) + b[l]  # 得到每一层的yk为一个向量，其中的值代表每一层节点的值
            yk = list(map(Sigmoid, yk))  # 将对应的值取Sigmoid
        ret.append(yk)
    print(ret)
def Update(weight_layers, bias_lyaers, D_weight_matrix, D_bias_matrix):
    return None
#Code View
w, b = MLP(2, 1, [2])
train_X = np.array([[0,0],[0,1],[1,0],[1,1]])
train_Y = np.array([[0], [1], [1], [0]])
w, b, lst = train(w, b, train_X, train_Y, 50000, 0.2)
test_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
predict(w,b,test_X)
import matplotlib.pyplot as plt
plt.plot(np.array(lst).T[0],np.array(lst).T[1])
plt.show()