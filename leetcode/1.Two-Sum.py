class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(nums)<2:
            return False
        test=dict()
        for i in range(0,len(nums)):
            if nums[i] in test:
                return [test[nums[i]],i]
            else:
                test[target-nums[i]]=i

so=Solution()
var=so.twoSum([2, 7, 11, 15],9)
print(var)
if 
    elif expression:
        pass