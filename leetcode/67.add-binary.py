class Solution:
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        bigger = max(len(a), len(b))
        carry = 0
        result = ""
        for i in range(1, bigger+1):
            if i > len(a):
                tmp = int(b[-1*i])+carry
                if tmp == 2:
                    carry = 1
                    tmp = 0
                else:
                    carry = 0
            elif i > len(b):
                tmp = int(a[-1*i])+carry
                if tmp == 2:
                    carry = 1
                    tmp = 0
                else:
                    carry = 0
            else:
                tmp = int(a[-1*i])+int(b[-1*i])+carry
                if tmp == 2:
                    carry = 1
                    tmp = 0
                elif tmp == 3:
                    carry = 1
                    tmp = 1
                else:
                    carry = 0
            result = str(tmp) + result
        if carry:
            result = str(carry) + result
        return result


if __name__ == "__main__":
    res = Solution().addBinary("11", "1")
    print(res)
