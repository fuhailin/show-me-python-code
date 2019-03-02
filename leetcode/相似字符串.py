#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re

#请完成下面这个函数，实现题目要求的功能
#当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^ 
#******************************开始写代码******************************
def str_info(data):
    res = []
    res_str = []
    for i  in range(0,len(data)):
        if data[i]  not in res_str:
            res_str.append(data[i])
            res.append(i)
        else:
            res_str.append(data[i])
            res.append(data.index(data[i]))
            
    return res

def  solve(S, T):
    t_res = str_info(T)

    count = 0

    for i in range(0,len(S)-len(T)+1):
        combine = S[i:i+len(T)]
        s_res = str_info(combine)
        if s_res == t_res:
            count +=1
            
    return count


#******************************结束写代码******************************

try:
    _S = input()
except:
    _S = None

try:
    _T = input()
except:
    _T = None
  
res = solve(_S, _T)
#res = solve("ababcb", "xyx")


print(str(res) + "\n")
