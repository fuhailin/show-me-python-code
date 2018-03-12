class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x<0: return False
        reverse=0
        ora=x
        while(x!=0):
            a=x%10
            reverse=reverse*10+a
            x=x//10
        if reverse==ora:
            return True
        else:
            return False


so=Solution()
var=so.isPalindrome(1221)
print(var)