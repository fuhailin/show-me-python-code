#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
import sys

def whether(c):
    for i in range(c+1):
        for j in range(i+1,c+1):
            tmp = i**4+j**4
            if tmp==c:
                return 1
    return 0

if __name__ == "__main__":
    # 读取第一行的n
    c = int(sys.stdin.readline().strip())

    ans = whether(c)
    
    print(ans)
    