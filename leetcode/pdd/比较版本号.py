import sys

def compare(ver1, ver2):
    list1 = ver1.split('.')
    list2 = ver2.split('.')
    length1 = len(list1)
    length2 = len(list2)
    maxlength = max(length1, length2)
    tmpList1 = [0 for i in range(maxlength)]
    tmpList2 = [0 for i in range(maxlength)]
    for i in range(length1):
        tmpList1[i] = int(list1[i])
    for j in range(length2):
        tmpList2[j] = int(list2[j])
    for k in range(maxlength):
        if tmpList1[k] > tmpList2[k]:
            return 1
        elif tmpList1[k] < tmpList2[k]:
            return -1
    return 0

if __name__ == '__main__':
    ver1 = sys.stdin.readline().strip()
    ver2 = sys.stdin.readline().strip()
    res = compare(ver1, ver2)
    print(res)