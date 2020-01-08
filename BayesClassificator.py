import numpy as np 
import pandas as pd
import math

def NaiveBayesMethod(dataSet):      # 朴素贝叶斯方法
    attrSet = dataSet.columns[:-1]  # 获得数据集的特征列
    labelTag = dataSet.columns[-1]  # 获得数据集的label
    attrProbabilityMap = {}         # 构建特征-概率图
    labelCount = dict(dataSet.iloc[:, -1].value_counts())  # 统计label的数目
    labelProba = dict(labelCount)
    #for label in labelProba: labelProba[label] = labelProba[label] / len(dataSet)  # 计算不同label的占比
    # ⬆ 为原计算方法
    #Laplace修正
    for label in labelProba: labelProba[label] = (labelProba[label] + 1) / (len(dataSet) + len(labelCount))  # 计算不同label的占比
    
    for attrs in attrSet:
        if (dataSet[attrs].dtypes == object):#属性值为object(str)
            attrProbabilityMap[attrs] = {}  #生成条件属性值对应字典
            attrSet = set(dataSet[attrs])
            for value in attrSet:           #对于属性的每一个可取值
                for label in labelProba:    # 将每个特征取值的对应label都计算一遍
                    #attrProbabilityMap[attrs][value + '|' + label] = len(dataSet[(dataSet[attrs] == value) & (dataSet[labelTag] == label)]) / labelCount[label]
                    # ⬆ 为原计算方法
                    #使用Laplace修正
                    attrProbabilityMap[attrs][value + '|' + label] = (len(dataSet[(dataSet[attrs] == value) & (dataSet[labelTag] == label)]) + 1) / (labelCount[label] + len(attrSet))
        
        elif (dataSet[attrs].dtypes == np.float64):  #属性为数值型
            attrProbabilityMap[attrs] = {}
            for label in labelProba:        # 对于数值型数据的每一个label都计算一次
                attrProbabilityMap[attrs][label] = [dataSet.loc[dataSet[labelTag] == label,attrs].mean(),dataSet.loc[dataSet[labelTag] == label,attrs].var()**0.5]# 将概率图对应于原数据的均值、标准差
    return attrProbabilityMap,labelProba    # 返回条件属性字典和label占比字典

def CheckNaiveBayes(check,attrProMap,labelProba):   # 由朴素贝叶斯条件属性字典和label占比字典预测
    ret = []
    attrSet = check.columns
    for index in range(len(check)):
        label_probability = dict(labelProba)        # 由label占比字典开始进行概率计算，可以通过使用对数来将乘法转为加法
        for attrs in attrSet:
            if (attrs in attrProMap and check[attrs].dtypes == object):
                for label in labelProba:
                    #print(attrs,label,attrProMap[attrs][check[attrs][index] + '|' + label])
                    label_probability[label] *= attrProMap[attrs][check[attrs][index] + '|' + label]
            elif (attrs in attrProMap and check[attrs].dtypes == np.float64):
                for label in labelProba:
                    mean, var = attrProMap[attrs][label]
                    #print(attrs,label,1/(((2*math.pi)**0.5)*var) * math.exp(-1*(check[attrs][index] - mean)**2 / (2 * (var**2))))
                    label_probability[label] *= 1/(((2*math.pi)**0.5)*var) * math.exp(-1*(check[attrs][index] - mean)**2 / (2 * (var**2)))
        ret.append(label_probability)
    return ret

#Code View
# data = pd.read_excel('data3.xls', encoding='gbk')   #读取训练数据
# attrProMap,labelProba = NaiveBayesMethod(data)      #生成朴素贝叶斯条件概率
# check = pd.read_excel('BayesCheck.xls', encoding='gbk')  #读取测试数据
# print(CheckNaiveBayes(check,attrProMap,labelProba)) #计算概率


# 贝叶斯网络计算 