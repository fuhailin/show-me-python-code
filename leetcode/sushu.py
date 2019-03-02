# -*- coding:utf-8 -*-  
import array,time
from math import log

def FindPrime(N):
    MAXNUM=int((log(N,10)*2.5+0.5)*N)
    i=2
    a=array.array('i')
    p=array.array('i')
    for x in range(MAXNUM):
        a.append(1)
    while(i<len(a)-1 and len(p)<N):
        if a[i]==1:
            p.append(i)
        for j in range(len(p)):
            if i*p[j]>=MAXNUM: break
            a[i*p[j]]=0
            if i%p[j]==0: break
        i+=1
    return(p)

def main():
    while True:
        N=input('输入需要找第几个素数：')
        N=int(N)
        mem=(log(N,10)*2.5+0.5)*N*4.3/1024/1024
        print("预计内存占用超过%0.2fMB！确定要继续吗？(y/n)" % mem)
        if input()!='y':
            continue
        start=time.time()
        p=FindPrime(N)
        print("找到啦：",p[-1])
        print("耗时：",time.time()-start,"秒")

if __name__=="__main__":
    main()