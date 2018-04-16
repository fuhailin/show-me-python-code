class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = list(set(nums))
        return len(nums)


def stringToIntegerList(input):
    return json.loads(input)


def integerListToString(nums, len_of_list=None):
    if not len_of_list:
        len_of_list = len(nums)
    return json.dumps(nums[:len_of_list])


def main():
    import sys
    def readlines():
        for line in sys.stdin:
            yield line.strip('\n')

    lines = readlines()
    while True:
        try:
            line = next(lines)
            nums = stringToIntegerList(line);

            ret = Solution().removeDuplicates(nums)

            out = integerListToString(nums, len_of_list=ret)
            print(out)
        except StopIteration:
            break


if __name__ == '__main__':
    main()