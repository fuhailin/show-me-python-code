class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        result = []
        tmp = ""
        for i in digits:
            tmp = tmp + str(i)
        tmp = str(int(tmp)+1)
        for i in range(len(tmp)):
            result.append(int(tmp[i]))
        return result