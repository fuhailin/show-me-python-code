def AppleDivide(n, array):
    if sum(array)%n != 0:
        return -1
    avg=sum(array)/n
    count=0
    for i in array:
        if (i-avg)%2!=0:
            return -1
        if i>avg:
            count+=(i-avg)
    return int(count/2)
n=int(input())
ls=list(map(int,input().split()))
print(AppleDivide(n,ls))