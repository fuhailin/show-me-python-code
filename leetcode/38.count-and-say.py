class Solution:
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        result = "1"
        for i in range(1, n):
            result = self.getNext(result)
        return result

    def getNext(self, last):
        result = str()
        count = 1
        i = 0
        while i < len(last):
            # for i in range(len(last), 1):
            if i == (len(last)-1):
                result = result+str(count)+last[i]
                break
            while last[i] == last[i+1]:
                count += 1
                i += 1
                if(i+1 == len(last)):
                    break
            result = result+str(count)+last[i]
            count = 1
            i += 1
        return result


test = Solution().countAndSay(2)
print(test)
