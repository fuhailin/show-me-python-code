import numpy as np


def get_max_drawdown_fast(array):
    drawdowns = []
    max_so_far = array[0]
    for i in range(len(array)):
        if array[i] > max_so_far:
            drawdown = 0
            drawdowns.append(drawdown)
            max_so_far = array[i]
        else:
            drawdown = max_so_far - array[i]
            drawdowns.append(drawdown)
    return max(drawdowns)


if __name__ == "__main__":
    test = [10, 11, 12, 13, 14, 1, 2, 3, 4, 5]
    #test1 = get_max_drawdown_fast(test)

    a=test.index(min(test))
    # 回撤结束时间点
    i = np.argmax(np.maximum.accumulate(test) - test)
    # 回撤开始的时间点
    j = np.argmax(test[:i])
    test1 = (float(test[i]) / test[j]) - 1.
    print(test1)
