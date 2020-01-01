import matplotlib.pyplot as plt
import numpy as np
import random
import math

def distance(pointA, pointB):       #计算欧式距离
    if (len(pointA) != len(pointB)):
        raise Exception("Two points is incompatible.")
    else:
        return sum([(pointA[index] - pointB[index])** 2 for index in range(len(pointA))])** 0.5 
def checkCore(core_list, point):  #根据聚类中心确定聚类核心
    min_distance = float('inf')
    min_class = 0
    for core in core_list:
        dis = distance(core, point)
        if (dis < min_distance):
            min_distance = dis
            min_class = core
    return min_class
def updateCore(cluster):       #根据聚类均值重新计算聚类中心
    new_core = set()
    for core in cluster:
        new_core.add((sum(a[0] for a in cluster[core])/len(cluster[core]),sum(a[1] for a in cluster[core]) / len(cluster[core])))   #计算平均向量
    return new_core
def KMeanscluster(pointData, K):
    core_clusters = set(random.sample(pointData, K))  #KMeans的初始聚类核心
    continue_flag = True
    while (continue_flag):
        clusters_dict = {core: [] for core in core_clusters}    #由当前的聚类中心生成聚类
        for point in pointData: #将所有的点进行聚类划分
            clusters_dict[checkCore(core_clusters, point)].append(point)
        new_cores = updateCore(clusters_dict)   #由当前的聚类平均重新得到聚类中心
        if (new_cores == core_clusters):    
            continue_flag = False  #若平均后得到的聚类中心未发生变化则KMeans聚类中心已经稳定，暂停循环
        core_clusters = set(new_cores)
    return clusters_dict

#测试代码
# dataPoint = [(random.random()*5+1,random.random()*5+2) for i in range(50)] + [(random.random()*5+3,random.random()*5+4) for i in range(50)]
# cluster = KMeanscluster(dataPoint,10)
# color = ["#000000","#FFFF00","#008000","#0000FF","#800080","#FFC0CB"]
# index = 0
# for tag in cluster:
#     plt.scatter(tag[0],tag[1],c="#F00000")
#     for p in cluster[tag]:
#         plt.scatter(p[0], p[1], c=color[index%len(color)])
#     index += 1
# plt.show()