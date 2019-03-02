import sys


def factorial(n):
    if(n > 1):
        return n * factorial(n - 1)
    else:
        return 1


if __name__ == "__main__":
    # number = sys.stdin.readline().strip()
    number = 20
    res=0
    for i in range(1, number+1):
        res+=factorial(i)
    print(res)
