class Solution:
    def charPiece(self,n):
        """
        :type n: string
        :rtype :int
        """
        result=n[0]
        for i in range(1,len(n)):
            temp=result[len(result)-1]
            if temp!=n[i]:
                result=result+n[i]
        return result

n = input()
# n="aaabbaaac"
so=Solution()
var=so.charPiece(n)
print('{:.2f}'.format(len(n)/len(var)))