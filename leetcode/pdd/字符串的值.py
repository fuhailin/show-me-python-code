# 已知一个由小写英文字母a-z组成的字符串，分别统计其中各个英文字母出现的次数，该字符串s的价值分为所有字母出现次数的平方和。 现在你可以将其中的一个字母全部换成另一个任意的字母，修改后字符串的最大价值为多少？
import sys


def getValue(string):
    if len(string) == 0:
        return 0
    if len(string) == 1:
        return 1
    mydic = {}
    for each in string:
        if each in mydic:
            mydic[each] += 1
        else:
            mydic[each] = 1
    mylist = list(mydic.values())
    mylist.sort()
    if len(mylist) == 1:
        return pow(mylist[0], 2)
    minV = mylist[len(mylist)-2]
    maxV = mylist[len(mylist)-1]
    res = pow(maxV+minV, 2)

    for i in range(0, len(mylist)-2):
        res = res + pow(mylist[i], 2)
    return res


if __name__ == '__main__':
    string = sys.stdin.readline().strip()

    res = getValue(string)
    print(res)
