#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re

#请完成下面这个函数，实现题目要求的功能
#当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^ 
#******************************开始写代码******************************


def  topk(nums):
    if len(nums)==0:
        return 0
    if len(nums)==1:
        return nums[0]
    d={}
    for i in nums:
        if i in d:
            d[i]+=1
        else:
            d[i]=1
    s = [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]
    index=0
    res=0
    for k, v in s:
        res+=k
        index+=1
        if index==2:
            return res


#******************************结束写代码**********for k, v in s:********************


_nums_cnt = 0
_nums_cnt = int(input())
_nums_i=0
_nums = []
while _nums_i < _nums_cnt:
    _nums_item = int(input())
    _nums.append(_nums_item)
    _nums_i+=1

  
res = topk(_nums)

print(str(res) + "\n")