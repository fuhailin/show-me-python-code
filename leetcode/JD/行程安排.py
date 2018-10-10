import os
import re

# 请完成下面这个函数，实现题目要求的功能
# 当然，你也可以不按照下面这个模板来作答，完全按照自己的想法来 ^-^
# ******************************开始写代码******************************


def schedule(data):
    

    start = data[0]
    end = data[1]
    res = 0
    for i in range(2, len(data), 2):
        s = data[i]
        e = data[i+1]
        if(s >= start and e <= end):
            res += 1
        elif(s < start and e > end):
            res += 1
            start = s
            end = e
    return res


# ******************************结束写代码******************************


_data_cnt = 0
_data_cnt = int(input())
_data_i = 0
_data = []
while _data_i < _data_cnt:
    _data_item = float(input())
    _data.append(_data_item)
    _data_i += 1


res = schedule(_data)

print(str(res) + "\n")
