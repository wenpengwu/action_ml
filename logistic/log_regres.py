import numpy as np


def loadDataSet():
    data_mat, labels = [], []
    with open("testSet.txt") as f:
        for line in f.readlines():
            a, b, c = line.strip().split()
            data_mat.append([1.0, float(a), float(b)])
            labels.append(int(c))
    return data_mat, labels


def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


"全量梯度，缺点数据量大是计算是个问题"


def grad_ascent(X, Y):
    X = np.mat(X)
    Y = np.mat(Y).transpose()
    m, n = np.shape(X)
    alpha = 0.001
    epochs = 500
    weights = np.ones((n, 1))  # 未加偏置b
    for k in range(epochs):
        h = sigmoid(X * weights)
        error = Y - h
        weights = weights + alpha * X.transpose() * error
    return weights.getA()


"随机梯度下降，这里的随机不是指的样本是随机的"


def stoc_gred_asent0(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    m, n = X.shape
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(np.sum(X[i] * weights))
        error = Y[i] - h
        weights = weights + alpha * error * X[i]
    return weights


"改进随机梯度算法"


def stoc_gred_acent1(X, Y, iter_num=150):
    X = np.array(X)
    Y = np.array(Y)
    m, n = X.shape
    weights = np.ones(3)
    for j in range(iter_num):
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.01
            rand_inx = int(np.random.uniform(0, m))
            h = sigmoid(np.sum(X[rand_inx, :] * weights))
            error = Y[rand_inx] - h
            weights = weights + alpha * error * X[rand_inx]
    return weights


def plot_best_fit(wei):
    import matplotlib.pyplot as plt

    weights = wei
    X, Y = loadDataSet()
    data_arr = np.array(X)
    n, _ = data_arr.shape
    xcord1, ycord1, xcord2, ycord2 = [], [], [], []
    for i in range(n):
        if int(Y[i] == 1):
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    plt.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    plt.scatter(xcord2, ycord2, s=30, c="green")
    X = np.arange(-3.0, 3.0, 0.1)
    w1, w2, w3 = weights
    plt.plot(X, (-w1 - w2 * X) / w3)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


"horse"


def classifyVec(X, weights):
    prob = sigmoid(np.sum(X * weights))
    return 1 if prob > 0.5 else 0


def colic_test():
    train_x, train_y = [], []
    with open("horseColicTraining.txt") as f:
        for line in f.readlines():
            *x, y = line.strip().split("\t")
            train_x.append(x)
            train_y.append(y)
    weights = stoc_gred_acent1(train_x, train_y, 500)
    errors, test_total = 0, 0.0
    with open("1horseColicTest.txt1") as f:
        for line in f.readlines():
            test_total += 1.0
            *x, y = line.strip().split("\t")
            if int(classifyVec(x)) != int(y):
                errors += 1
    rate = float(errors / test_total)
    print("the rate is:%f" % rate)
    return rate


def multi_test():
    epochs = 10
    total_error = 0.0
    for k in range(epochs):
        total_error += colic_test()
    print(
        "after %d iterations the average error is :%f"
        % (epochs, total_error / float(epochs))
    )


if __name__ == "__main__":
    X, Y = loadDataSet()
    # wei = grad_ascent(X,Y)
    wei = stoc_gred_acent1(X, Y)
    plot_best_fit(wei)
