# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        jump=0
        if number==1:
            jump=jump+1
        else:
            jump=jump+2*self.jumpFloorII(number-1)
        return jump

so=Solution()
var=so.jumpFloorII(2)
print(var)