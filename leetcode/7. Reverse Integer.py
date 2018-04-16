class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign=1
        if x<0:
           sign=-1
        x=x*sign
        result=0
        while(x!=0):
            a=x%10
            x=x//10
            result=result*10+a
            
        if abs(result) > 0x7FFFFFFF:
            return 0
        else:
            return sign*result
so=Solution()
var=so.reverse(900000)
print(var)