'''
一座城市有若干栋高楼，为举行一项特殊的定高跳极的活动需要N栋楼高度相同。
市长有权力对任何高楼增加若干层，求一共最少需要增加多少层楼才能满足举办活动的条件。
'''

import sys


def getValue(mylist, num):
    res = 0
    mydic = {}
    for each in mylist:
        if each in mydic:
            mydic[each] += 1
        else:
            mydic[each] = 1

    sorted_d = sorted((value, key) for (key,value) in mydic.items())
    return res


if __name__ == '__main__':
    list1 = [int(i) for i in sys.stdin.readline().strip().split(' ')]
    list2 = [int(i) for i in sys.stdin.readline().strip().split(' ')]

    res = getValue(list2, list1[1])
    print(res)
