class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tmp=set(nums)
        nums = list(tmp)
        return len(nums)

so=Solution()
var=so.removeDuplicates([1,1,2])
print(var)