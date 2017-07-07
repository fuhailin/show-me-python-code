#!/usr/bin/env python
import numpy as np
import math

# ������numpy��python-Levenshtein��scipy


def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())
# euclidean,ŷʽ�����㷨���������Ϊ��������������ֵΪŷʽ����


def Manhattan(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return np.abs(npvec1-npvec2).sum()
# Manhattan_Distance,�����پ���


def Chebyshev(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return max(np.abs(npvec1-npvec2))
# Chebyshev_Distance,�б�ѩ�����


from math import*
from decimal import Decimal

def nth_root(value,n_root):
    root_value=1/float(n_root)
    return round(Decimal(value)**Decimal(root_value),3)

def minkowski_distance(vec1, vec2,params):
    return nth_root(sum(pow(abs(a-b),params) for a,b in zip(vec1, vec2)),params)

print(minkowski_distance([0,3,4,5],[7,6,3,-1],3))


def Standardized_Euclidean(vec1, vec2, v):
    from scipy import spatial
    npvec = np.array([np.array(vec1), np.array(vec2)])
    return spatial.distance.pdist(npvec, 'seuclidean', V=None)
# Standardized Euclidean distance,��׼��ŷ�Ͼ���
# �ڶԳ�����������о����ʱ����ͨ�ľ����޷�����Ҫ��
# ������ͨ�ľ��������Ĵ����Բ�ε�������ʱ��Ҫ���ñ�׼��ŷʽ���롣
# �ο�  ��׼��ŷʽ���룺http://blog.csdn.net/jinzhichaoshuiping/article/details/51019473

def Mahalanobis(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    npvec = np.array([npvec1, npvec2])
    sub = npvec.T[0]-npvec.T[1]
    inv_sub = np.linalg.inv(np.cov(npvec1, npvec2))
    return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))
# MahalanobisDistance,���Ͼ���


def Edit_distance_str(str1, str2):
    import Levenshtein
    edit_distance_distance = Levenshtein.distance(str1, str2)
    similarity = 1-(edit_distance_distance/max(len(str1), len(str2)))
    return {'Distance': edit_distance_distance, 'Similarity': similarity}
# Levenshtein distance,�༭���룬���ڼ��������ַ���֮��ı༭���룬�������Ϊ�����ַ���


def Edit_distance_array(str_ary1, str_ary2):
    len_str_ary1 = len(str_ary1) + 1
    len_str_ary2 = len(str_ary2) + 1
    matrix = [0 for n in range(len_str_ary1 * len_str_ary2)]
    for i in range(len_str_ary1):
        matrix[i] = i
    for j in range(0, len(matrix), len_str_ary1):
        if j % len_str_ary1 == 0:
            matrix[j] = j // len_str_ary1
    for i in range(1, len_str_ary1):
        for j in range(1, len_str_ary2):
            if str_ary1[i-1] == str_ary2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[j*len_str_ary1+i] = min(matrix[(j-1)*len_str_ary1+i]+1, 
            matrix[j*len_str_ary1+(i-1)]+1, matrix[(j-1)*len_str_ary1+(i-1)] + cost)
    distance = int(matrix[-1])
    similarity = 1-int(matrix[-1])/max(len(str_ary1), len(str_ary2))
    return {'Distance': distance, 'Similarity': similarity}
# ����б��д�ı༭���룬��NLP�����У����������ı������ƶȣ��ǻ��ھ����дʺʹ�֮��Ĳ��졣
# ���ʹ�ô�ͳ�ı༭�����㷨��������Ϊ�ı���������֮��ı༭������������ݱ༭�����˼ά��
# ���༭�����еĴ����ַ����е��ַ����󣬱�ɴ���list��ÿ��Ԫ��


def Cosine(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))
# Cosine�����Ҽн�
# ����ѧϰ�н�����һ������������������֮��Ĳ��졣
# Ҳ����ʹ�����������ƶ��㷨��