#! python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#指定默认字体，以便matplotlib在图中输出中文
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 


#尽管以下这句引用没有被直接使用，它是为了能让3D投影正常工作
from mpl_toolkits.mplot3d import Axes3D

#引用sklearn聚类模型中的KMeans
from sklearn.cluster import KMeans
#引用sklearn收集好的数据集
from sklearn import datasets

#固定随机数种子，如果没有它每次运行的结果都将是不同的随机结果
np.random.seed(5)

#加载鸢尾花卉数据集Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

#这里我们通过初始化三个不同的KMeans聚类模型，比较聚类结果的影响
#减少算法用不同质心种子运行的次数，默认的结果是连续运行10次的最佳输出，
#第三个模型中我们将“n_init”固定为1，所以这个模型的初始随机质心将会较差
estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,init='random'))]

fignum = 1
titles = ['8个簇', '3个簇', '3个簇, 随机初始化较差']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('花瓣宽度')
    ax.set_ylabel('萼片长度')
    ax.set_zlabel('花瓣长度')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1


print(0)