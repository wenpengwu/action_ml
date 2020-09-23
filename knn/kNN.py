import numpy as np
import collections


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classfiy0(input, dateSet, labels, k):
    diffMat = (input - dateSet) ** 2
    distances = (np.sum(diffMat, axis=-1)) ** 0.5
    sortedDistances = np.argsort(distances)  # 返回下标
    inx = sortedDistances[:k]
    return collections.Counter(labels[inx]).most_common(1)[0][0]


def file2matrix(filename):
    with open(filename) as f:
        fr = f.readlines()
        returnMatrix = np.zeros((len(fr), 3))
        labels = []
        i = 0
        for line in fr:
            line = line.strip().split("\t")
            returnMatrix[i, :] = line[0:3]
            labels.append(int(line[-1]))
            i += 1
    return returnMatrix, np.array(labels)


def autoNorm(dateSet):
    minVals = dateSet.min(0)
    maxVals = dateSet.max(0)
    ranges = maxVals - minVals
    dateSet -= minVals
    dateSet /= ranges
    return dateSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.1
    datas, labels = file2matrix("datingTestSet2.txt")
    normat, ranges, mins = autoNorm(datas)
    m = normat.shape[0]
    errors = 0
    numTestVecs = int(hoRatio * m)
    for i in range(numTestVecs):
        result = classfiy0(
            normat[i, :], normat[numTestVecs:, :], labels[numTestVecs:], 3
        )
        print("预测值:%d,真实值:%d" % (result, labels[i]))
        if result != labels[i]:
            errors += 1
    print("错误率%f" % (errors / float(numTestVecs)))


def img2vector(filename):
    returnVec = np.zeros((1, 1024))
    with open(filename) as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                returnVec[0, 32 * i + j] = int(line[j])
    return returnVec


def handwritingClassTest():
    train_y = []
    from os import listdir

    train_list = listdir("trainingDigits")
    m = len(train_list)
    train_x = np.zeros((m, 1024))
    for i in range(m):
        file_name = train_list[i]
        label = int(file_name.split(".")[0].split("_")[0])
        train_y.append(label)
        train_x[i, :] = img2vector("trainingDigits/%s" % file_name)
    test_list = listdir("testDigits")
    eorros = 0.0
    m = len(test_list)
    for i in range(m):
        file_name = train_list[i]
        y_true = int(file_name.split(".")[0].split("_")[0])
        test_x = img2vector("trainingDigits/%s" % file_name)
        y_pred = classfiy0(test_x, train_x, train_y, 3)
        print("真实值:%d,预测值:%d" % (y_true, y_pred))
        if y_pred != y_true:
            eorros += 1
    print("预测错误率:%f" % (eorros / float(m)))


if __name__ == "__main__":
    """
    group, labels = createDataSet()
    print(classfiy0(np.array([0,0]),group,np.array(labels),3))    
    """
    """
    """
    datingClassTest()
    exit(0)
    "绘图"
    # import matplotlib.pyplot as plt
    # plt.scatter(datas[:,1],datas[:,2],15.*np.array(labels),15*np.array(labels))
    # plt.show()
