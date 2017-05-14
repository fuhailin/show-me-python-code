#! python3
# -*- coding: utf-8 -*-
import re
import pandas as pd


def Predict(userdata, ruleframe):
    result = dict()
    for line in ruleframe.itertuples():
        a = re.findall(r"\d+\.?\d*", line[1])
        ruleset = set(map(int, set(a)))
        test3 = userdata & ruleset
        if len(test3) > 0:
            b = re.findall(r"\d+\.?\d*", line[2])
            c = float(b[-1:][0])  # 置信度
            d = b[:-1]
            result[str(d)] = c
    return result


if __name__ == '__main__':
    ruleframe = pd.read_table('rules.txt', sep='==>', header=None, engine='python')  # 输出规则
    traindata = pd.read_table('train5_1.txt', sep=' ', header=None)  # 训练数据集
    testdata = pd.read_table('validation5_1.txt', sep=' ', header=None)  # 测试数据集
    num_row = traindata.shape[0]
    allruleset = set(map(int, re.findall(r"\d+\.?\d*", str(ruleframe.icol(0).values))))  # 所有输出规则集
    # trueradio = []# 真实结果集
    preradio = []  # 所有预测结果集
    hit = []  # 命中集
    for i in range(0, num_row):  # 逐行预测
        userset = set(traindata.irow(i).values)  # 用户历史记录集合
        test = userset & allruleset  # 用户历史记录集合与规则集合的交集
        if len(test) > 0:  # 如果rules集合中包含用户元素，然后根据置信度进行预测
            res = Predict(userset, ruleframe)  # 将rules文件中包含用户元素的规则找出
            # trueradio.append(truth)
            print(i)  # 程序运行指示
            maxsupport = 0
            for items, support in res.items():
                if support > maxsupport:
                    predict_items = items  # 找到最大支持度对应的规则预测项
                    maxsupport = support  # 找到最大支持度
            predict_item = int(re.findall(r"\d+\.?\d*", predict_items)[0])
            preradio.append(predict_item)
            truth = testdata.irow(i).values[0]  # 真实结果
            if predict_item == truth:  # 判断预测是否准确
                hit.append(predict_item)
    print(0)
    precision = len(hit) / len(preradio)
    print(precision)
