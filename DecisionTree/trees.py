from math import log
from collections import defaultdict


def calcShannonEnt(dateSet):
    labelCounts = defaultdict(int)
    total = len(dateSet)
    for vec in dateSet:
        label = vec[-1]
        labelCounts[label] += 1
    Ent = 0.0
    for k in labelCounts:
        prob = float(labelCounts[k]) / total
        Ent -= prob * log(prob, 2)
    return Ent


def splitDataSet(dataSet, axis, value):
    ret = []
    for vec in dataSet:
        if vec[axis] == value:
            tmp = vec[:axis]
            tmp.extend(vec[axis + 1 :])
            ret.append(tmp)
    return ret


"""
划分之后的集合求加权熵和 与基础集合中熵的差值称为信息增益
"""


def chooseBestSplit(dateSet):
    features_num = len(dateSet[0]) - 1
    baseEnt = calcShannonEnt(dateSet)
    bestInfoGain, bestFeature = 0.0, -1

    for i in range(features_num):
        featList = [a[i] for a in dateSet]  # 取得某列的值
        uniq_vals = set(featList)
        newEnt = 0.0
        for v in uniq_vals:
            subdataset = splitDataSet(dateSet, i, v)
            prob = len(subdataset) / float(len(dateSet))
            newEnt += prob * calcShannonEnt(subdataset)
        infoGain = baseEnt - newEnt

        if bestInfoGain < infoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(class_list):
    class_count = defaultdict(int)
    for vote in class_list:
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]
