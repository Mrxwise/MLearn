import pandas as pd
import math

def getMostLabel(data_list):
    return data_list.value_counts().idxmax()    #DataFrame中，value_counts返回元素计数Series,idxmax返回最大值元素

def Ent(data):                                  #计算信息熵，一般依据label列进行信息熵的计算
    value_count = dict(data.iloc[:,-1].value_counts())
    ent = 0
    for value in value_count:
        P = value_count[value] / len(data)
        ent += P * math.log2(P)
    return 0 - ent
    
def getGain(col, data):  #计算信息增益
    # data.groupby(col) 将data按照col进行分组，即在data中col属性相同的分为一组
    # 同时对data.groupby().appy(lambda x: fun(x)) 相当对每个分组x进行fun(x)操作
    subGain = data.groupby(col).apply(lambda x: Ent(x) * len(x) / len(data)).sum()  #求得划分后每个分组的Ent值的和
    return Ent(data) - subGain  #得到划分后的信息增益
    
def getMaxGain(data_set):   #由信息增益进行划分
    max_gain = -float('inf')
    max_label = None
    for col in data_set.columns[:-1]:
        gain = getGain(col, data_set)   #计算当前col的信息增益
        #print(col,gain)
        if gain > max_gain:             #判断最大信息增益值
            max_gain = gain
            max_label = col
    return max_label

def DTree(dataSet):
    if ('编号' in dataSet.columns[:-1]):
        dataSet = dataSet.drop('编号',1)
    attrSet = dataSet.columns[:-1]  #得到目前数据集中的属性集
    labelList = dataSet.iloc[:,-1]  #得到数据集中最后一列
                                    #iloc[n] 按列选择数据

    #DT使用字典的形式进行存储
    if (len(pd.unique(labelList)) == 1): #dataSet中所有样本属于同一个类别C，将此节点标记为C类别 ==>情形1
        #print(labelList.iloc[0])
        return labelList.iloc[0]  #pandas.unique(Series) 返回除重后的Series
    
    if (len(attrSet) == 0): #dataSet中带划分的属性集为空    ==>对应于情形2
        return getMostLabel(labelList)  #选择LabelList中选择数量最多的label
    if (len(dataSet.loc[:, attrSet].drop_duplicates()) == 1):  #dataSet中对应于labelList的特征都相同
        #用标签选取多列数据使用 df.loc[:,['A','B']].iloc[0] ->选取切分的'A','B'的第零行
        return getMostLabel(labelList)  #选择LabelList中数量最多的label
    
    #对应于情形3，计算最优属性集
    best_attr = getMaxGain(dataSet)     #得到最优划分 
    subTree = {best_attr: {}}

    for newAttr, split_data in dataSet.groupby(best_attr):     #依照最优属性中每一个分组值进行划分
        #其中newAttr 为每一个分组中的best_attr的值，即新的划分值。split_data为对应分组的data
        if (len(split_data) == 0):              #新划分的数据为0,则依据总labelList，设置为最多的label值
            subTree[best_attr][newAttr] = getMostLabel(labelList)
        else:
            new_data = split_data.drop(best_attr,1)  #将划分的属性去掉 
            subTree[best_attr][newAttr] = DTree(new_data)  #同时在subTree[best_attr][newAttr]中即对应新生成的子树中再对数据进行划分
            
    return subTree


data = pd.read_excel('data.xls',encoding='gbk')
judge = DTree(data)
print(judge)

def DTpredict(tree, data):
    attr = list(tree.keys())[0]  # attr为本次搜寻的feature
    label = data[attr]
    next_step = tree[attr][label]
    if (type(next_step) == str):    #搜寻得到str的结果
        return next_step
    else:
        return DTpredict(next_step, data)  #继续在下一个子树中进行搜寻
        
# https://blog.csdn.net/sinat_38682860/article/details/82428674


#目前尚存在问题：
#1、基于连续值的DT以字典生成还存在问题
#2、由DT进行bagging进行集成学习可得到RF
