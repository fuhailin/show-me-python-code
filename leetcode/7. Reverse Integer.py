class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if abs(x) >= 2**31: return 0
        if x<0:
           sign=-1
        else:
            sign=1
        x=x*sign
        result=0
        while(x!=0):
            a=x%10
            x=x//10
            result=result*10+a
        return sign*result
so=Solution()
var=so.reverse(1534236469)
print(var)