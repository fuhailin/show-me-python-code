# coding: utf-8
'''
求两个字符串的最长公共子串
思路：建立一个二维数组，保存连续位相同与否的状态
'''


def longest_commom_substring(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    record = [[0 for x in range(len_str2+1)]
              for y in range(len_str1+1)]  # Why 多一位
    maxNum = 0  # 最长匹配长度
    p = 0

    for i in range(len_str1):
        for j in range(len_str2):
            if(str1[i] == str2[j]):
                # 相同则累加
                record[i+1][j+1] = record[i][j] + 1
                if record[i+1][j+1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i+1][j+1]
                    # 记录最大匹配长度的最终位置
                    p = i+1
    return str1[p-maxNum:p], maxNum

def lcs(a,b):
	lena=len(a)
	lenb=len(b)
	c=[[0 for i in range(lenb+1)] for j in range(lena+1)]
	flag=[[0 for i in range(lenb+1)] for j in range(lena+1)]
	for i in range(lena):
		for j in range(lenb):
			if a[i]==b[j]:
				c[i+1][j+1]=c[i][j]+1
				flag[i+1][j+1]='ok'
			elif c[i+1][j]>c[i][j+1]:
				c[i+1][j+1]=c[i+1][j]
				flag[i+1][j+1]='left'
			else:
				c[i+1][j+1]=c[i][j+1]
				flag[i+1][j+1]='up'
	return c,flag

def printLcs(flag,a,i,j):
	if i==0 or j==0:
		return
	if flag[i][j]=='ok':
		printLcs(flag,a,i-1,j-1)
		print(a[i-1],end='')
	elif flag[i][j]=='left':
		printLcs(flag,a,i,j-1)
	else:
		printLcs(flag,a,i-1,j)


if __name__ == "__main__":
    # str1 = input()
    # str2 = input()
    str1 = "ababcb"
    str2 = "aba"
    c,flag=lcs(str1,str2)
    printLcs(flag,str1,len(str1),len(str2))
    print(longest_commom_substring(str1, str2))
