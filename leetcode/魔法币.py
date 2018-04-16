class Solution:
    def magiccoin(self,n):
        """
        :type n: int
        :rtype :string
        """
        temp=[]
        if n<=0:
            return 0
        while(n>0):
            if (n%2)==0:
                temp.append(2)
                n=(n-2)//2
            else:
                temp.append(1)
                n=(n-1)//2
        res=''
        for i in range(1,len(temp)+1):
            res=res+str(temp[len(temp)-i])
        return res
n = int(input())
so=Solution()
var=so.magiccoin(n)
print(var)