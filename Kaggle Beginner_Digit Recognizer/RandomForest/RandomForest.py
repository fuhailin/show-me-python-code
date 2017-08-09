from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

CPU = -1


def main():
    print("Reading training set")
    '''
    _dataset = genfromtxt('../data/train.csv', delimiter=',', dtype='int64')
    dataset=_dataset[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    print("Reading test set")
    test = genfromtxt('../data/test.csv', delimiter=',', dtype='int64')[1:]
    '''
    # create the training & test sets, skipping the header row with [1:]
    dataset = pd.read_csv("../data/train.csv")
    target = dataset['label'].values.ravel()
    train = dataset.iloc[:, 1:].values
    test = pd.read_csv("../data/test.csv").values

    # create and train the random forest
    # n_estimators：决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了）可以达到可接受的性能和误差率
    # n_jobs：并行job个数。这个在ensemble算法中非常重要，尤其是bagging（而非boosting，因为boosting的每次迭代之间有影响，所以很难进行并行化），
    # 因为可以并行从而提高性能。1=不并行；n：n个并行；-1：CPU有多少core，就启动多少job
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=CPU)
    print("Fitting RF classifier")
    rf.fit(train, target)

    print("Predicting test set")
    prediction=rf.predict(test)
    np.savetxt('../data/submission_RandomForest.csv', np.c_[range(1, len(test) + 1), prediction], delimiter=',', header='ImageId,Label', comments='', fmt='%d')


if __name__ == "__main__":
    main()
