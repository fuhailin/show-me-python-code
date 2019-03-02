#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re

#请完成下面这个函数，实现题目要求的功能
#当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^ 
#******************************开始写代码******************************


#******************************结束写代码******************************


# _p_cnt = 0
# _p_cnt = int(input())
# _p_i=0
# _p = []
# while _p_i < _p_cnt:
#     _p_item = int(input())
#     _p.append(_p_item)
#     _p_i+=1

# _M = int(input())

  
_p = [99,199,1999,10000,39,1499]
_M = 10238
# res=subset_sum(_p, _M)

import itertools

result = [seq for i in range(len(_p), 0, -1) for seq in itertools.combinations(_p, i) if sum(seq) == _M]
print(result)

print(str(int(res)) + "\n")