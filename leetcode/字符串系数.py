#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
import sys


def common_substr(a, b, k):
    res=0
    substrs = set(a[i:i+k] for i in range(len(a)-k+1))
    for substr in (b[i:i+k] for i in range(len(b)-k+1)):
        if substr in substrs:
            res+=1
    return res


if __name__ == "__main__":
    # 读取第一行的n
    k = int(sys.stdin.readline().strip())
    stringA = sys.stdin.readline().strip()
    stringB = sys.stdin.readline().strip()
    # k=2
    # stringA="abab"
    # stringB="ababab"
    # test = stringB.split('ab')
    ans = common_substr(stringA,stringB,k)
    
    print(ans)