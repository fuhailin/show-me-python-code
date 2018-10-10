#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
import sys

ASCII_SIZE = 256
 
def getMaxOccuringChar(str):
    count = [0] * ASCII_SIZE
 
    # Utility variables
    max = -1
    c = ''
 
    # Traversing through the string and maintaining the count of
    # each character
    for i in str:
        count[ord(i)]+=1;
 
    for i in str:
        if max < count[ord(i)]:
            max = count[ord(i)]
            c = i
 
    return c, max

if __name__ == "__main__":
    line = sys.stdin.readline().strip()
    line = line.upper()
    line = line.replace(' ', '')

    test, test1 = getMaxOccuringChar(line)
    # print(test)

    # res = {}
    # for i in line:
    #     if i in res:
    #         res[i]+=1
    #     else:
    #         res[i]=1
    
    # s = [(k, res[k]) for k in sorted(res, key=res.get, reverse=True)]
    # tmp = s[0]
    print(str(test)+str(test1))

    