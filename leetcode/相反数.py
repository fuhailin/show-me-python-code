class Solution:
    def reverseNum(self,n):
        """
        :type n: int
        :rtype :int
        """
        temp=n
        result=0
        while(n>0):
            a=n%10
            result=result*10+a
            n=n//10
        return result+temp
n = int(input())
so=Solution()
var=so.reverseNum(n)
print(var)