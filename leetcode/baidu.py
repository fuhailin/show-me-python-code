def countdiff(mystring):
    res = set()
    res.add(mystring)
    for i in range(len(mystring)):
        tmp = mystring[1:]
        mystring = mystring[1:]+mystring[0]
        res.add(mystring)
    return len(res)

if __name__ == "__main__":
    # test = input()
    test = "ABABA"
    print(countdiff(test))