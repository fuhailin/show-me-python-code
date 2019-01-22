class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        wordlist = s.rstrip(' ').split(' ')
        if len(wordlist)==0:
            return 0
        return len(wordlist[-1])