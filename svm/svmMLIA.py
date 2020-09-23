import numpy as np

"投影距离"


"""

W^Tx+b  W的转置乘以X + b

拉格朗日数乘法
"""


def load_dataset(filename):
    x_data, y_data = [], []
    with open(filename) as f:
        for line in f.readlines():
            a, b, c = line.strip().split("\t")
            x_data.append([float(a), float(b)])
            y_data.append(float(c))
    return x_data, y_data


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"简化版SMO"


def smo_simple(x_data, y_data, C, toler, iter_num):
    x_data = np.mat(x_data)
    y_data = np.mat(y_data).transpose()
    b = 0
    m, n = x_data.shape
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < iter_num:
        alpha_pairs_changed = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, y_data).T * (x_data * x_data[i, :].T)) + b
            ei = fxi - float(y_data[i])
            if (y_data[i] * ei < -toler and alphas[i] < C) or (
                y_data[i] * ei > toler and alphas[i] > 0
            ):
                j = select_jrand(i, m)
                fxj = (
                    float(np.multiply(alphas, y_data).T * (x_data * x_data[j, :].T)) + b
                )
                ej = fxj - float(y_data[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if y_data[i] != y_data[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue

                eta = (
                    2.0 * x_data[i, :] * x_data[j, :].T
                    - x_data[i, :] * x_data[i, :].T
                    - x_data[j, :] * x_data[j, :].T
                )
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= y_data[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += y_data[j] * y_data[i] * (alphaJold - alphas[j])
                b1 = (
                    b
                    - ei
                    - y_data[i]
                    * (alphas[i] - alphaIold)
                    * x_data[i, :]
                    * x_data[i, :].T
                    - y_data[j]
                    * (alphas[j] - alphaJold)
                    * x_data[i, :]
                    * x_data[j, :].T
                )
                b2 = (
                    b
                    - ej
                    - y_data[i]
                    * (alphas[i] - alphaIold)
                    * x_data[i, :]
                    * x_data[j, :].T
                    - y_data[j]
                    * (alphas[j] - alphaJold)
                    * x_data[i, :]
                    * x_data[j, :].T
                )
                if alphas[i] > 0 and C > alphas[i]:
                    b = b1
                elif alphas[j] > 0 and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter :%d i:%d. pairs change %d" % (iter, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iter += 1
        else:
            iter = 0
        print("tier number:%d" % iter)
    return b, alphas


class OptStruct:
    def __init__(self, input_mat, labels, C, toler):
        self.X = input_mat
        self.labels = labels
        self.C = C
        self.tol = toler
        self.m = input_mat.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))


def cal_ek(os, k):
    """

    Parameters
    ----------
    os:OptStruct
    k:int

    Returns ek:float
    -------

    """
    fxk = float(np.multiply(os.alphas, os.labels).T * (os.X * os.X[k, :].T)) + os.b
    ek = fxk - float(os.labels[k])
    return ek


def select_j(i, os, ei):
    """

    Parameters
    ----------
    i
    os:OptStruct
    ei

    Returns
    -------

    """
    max_k, max_detal_e, ej = -1, 0, 0
    os.e_cache[i] = [1, ei]
    valid_e_cache_list = np.nonzero(os.e_cache[:, 0].A)[0]
    if len(valid_e_cache_list) > 1:
        for k in valid_e_cache_list:
            if k == i:
                continue
            ek = cal_ek(os, k)
            delat_e = abs(ei - ek)
            if delat_e > max_detal_e:
                max_k = k
                max_detal_e = delat_e
                ej = ek
        return max_k, ej
    else:
        j = select_jrand(i, os.m)
        ej = cal_ek(os, j)
        return j, ej


def update_ek(os, k):
    """

    Parameters
    ----------
    os:OptStruct
    k

    Returns
    -------

    """
    ek = cal_ek(os, k)
    os.e_cache[k] = [1, ek]


def innerL(i, os):
    """

    Parameters
    ----------
    i
    os:OptStruct

    Returns
    -------

    """

    ei = cal_ek(os, i)
    if (os.labels[i] * ei < -os.tol and os.alphas[i] < os.C) or (
        os.labels[i] * ei > os.tol and os.alphas[i] > 0
    ):
        j, ej = select_j(i, os, ei)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if os.labels[i] != os.labels[j]:
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = (
            2.0 * os.X[i, :] * os.X[j, :].T
            - os.X[i, :] * os.X[i, :].T
            - os.X[j, :] * os.X[j, :].T
        )
        if eta >= 0:
            print("eta>=0")
            return 0
        os.alphas[j] -= os.labels[j] * (ei - ej) / eta
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)
        update_ek(os, j)
        if abs(os.alphas[j] - alphaJold) < 0.00001:
            print("j 移动不足")
            return 0
        os.alphas[i] += os.labels[j] * os.labels[i] * (alphaJold - alphaIold)
        update_ek(os, i)
        b1 = (
            os.b
            - ei
            - os.labels[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[i, :].T
            - os.labels[j] * (os.alphas[j] - alphaJold) * os.X[i, :] * os.X[i, :].T
        )
        b2 = (
            os.b
            - ej
            - os.labels[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[i, :].T
            - os.labels[j] * (os.alphas[j] - alphaJold) * os.X[i, :] * os.X[i, :].T
        )
        if os.alphas[i] and os.C > os.alphas[i]:
            os.b = b1
        elif os.alphas[j] > 0 and os.C > os.alphas[j]:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smop(x_data, labels, C, toler, iter_num, kTup=("lin", 0)):
    os = OptStruct(np.mat(x_data), np.mat(labels).transpose(), C, toler)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    while iter < iter_num and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(os.m):
                alpha_pairs_changed += innerL(i, os)
                print(
                    "fullset ,iter:%d i: %d,pairs changed %d"
                    % (iter, i, alpha_pairs_changed)
                )
            iter += 1
        else:
            # 返回ndarray matrix.A
            non_bound_ids = np.nonzero(os.alphas.A > 0)
            for i in non_bound_ids:
                alpha_pairs_changed += innerL(i, os)
                print(
                    "non-bound iterm:%d i:%d ,pair changed %d"
                    % (iter, i, alpha_pairs_changed)
                )
            iter += 1

        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print("iteration number:%d" % iter)
    return os.b, os.alphas


def cal_ws(alphas, x_data, labels):
    x_data = np.mat(x_data)
    labels = np.mat(labels).transpose()
    m, n = x_data.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labels[i], x_data[i, :].T)
    return w


"核转换函数"


def kernel_trans(X, A, kTup):
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == "lin":
        K = X * A.T
    elif kTup[0] == "rbf":
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = np.exp(K / -1 * kTup[1] ** 2)
    else:
        raise NameError("休斯顿我们有个问题--核方法无法识别")
    return K


class KernelOptStruct:
    def __init__(self, input_mat, labels, C, toler, kTup):
        self.X = input_mat
        self.labels = labels
        self.C = C
        self.tol = toler
        self.m = input_mat.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.e_cache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], kTup)


def kernel_cal_ek(os, i):
    """

    Parameters
    ----------
    os:KernelOptStruct
    i

    Returns
    -------

    """
    fxk = float(np.multiply(os.alphas, os.labels).T * os.K[:, i] + os.b)
    ek = fxk - float(os.labels[k])
    return ek


def kernel_innerL(os, i):
    """

       Parameters
       ----------
       i
       os:KernelOptStruct

       Returns
       -------

       """

    ei = kernel_cal_ek(os, i)
    if (os.labels[i] * ei < -os.tol and os.alphas[i] < os.C) or (
        os.labels[i] * ei > os.tol and os.alphas[i] > 0
    ):
        j, ej = select_j(i, os, ei)
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        if os.labels[i] != os.labels[j]:
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = (
            2.0 * os.X[i, :] * os.X[j, :].T
            - os.X[i, :] * os.X[i, :].T
            - os.X[j, :] * os.X[j, :].T
        )
        if eta >= 0:
            print("eta>=0")
            return 0
        os.alphas[j] -= os.labels[j] * (ei - ej) / eta
        os.alphas[j] = clip_alpha(os.alphas[j], H, L)
        update_ek(os, j)
        if abs(os.alphas[j] - alphaJold) < 0.00001:
            print("j 移动不足")
            return 0
        os.alphas[i] += os.labels[j] * os.labels[i] * (alphaJold - alphaIold)
        update_ek(os, i)
        b1 = (
            os.b
            - ei
            - os.labels[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[i, :].T
            - os.labels[j] * (os.alphas[j] - alphaJold) * os.X[i, :] * os.X[i, :].T
        )
        b2 = (
            os.b
            - ej
            - os.labels[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[i, :].T
            - os.labels[j] * (os.alphas[j] - alphaJold) * os.X[i, :] * os.X[i, :].T
        )
        if os.alphas[i] and os.C > os.alphas[i]:
            os.b = b1
        elif os.alphas[j] > 0 and os.C > os.alphas[j]:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def test_rbf(kl=1.3):
    x_data, y_data = load_dataset("testSetRBF.txt")
    b, alphas = smop(x_data, y_data, 200, 0.0001, 10000, ("rbf", kl))
    x_data, y_data = np.mat(x_data), np.mat(y_data).transpose()
    sv_inx = np.nonzero(alphas.A > 0)[0]
    sv_s = x_data[sv_inx]
    label_sv = y_data[sv_inx]
    print("there are %d support vectors" % sv_s.shape[0])
    m, n = x_data.shape
    errors = 0
    for i in range(m):
        kernel_e_val = kernel_trans(sv_s, x_data[i, :], ("rbf", kl))
        predict = kernel_e_val.T * np.multiply(label_sv, alphas[sv_inx]) + b
        if np.sign(predict) != np.sign(y_data[i]):
            errors += 1
    print("训练数据集错误率%f" % (float(errors) / m))
    errors = 0
    x_test, y_test = load_dataset("testSetRBF2.txt")
    x_test, y_test = np.mat(x_test), np.mat(y_test).transpose()
    m, n = x_test.shape
    for i in range(m):
        kernel_e_val = kernel_trans(sv_s, x_data[i, :], ("rbf", kl))
        predict = kernel_e_val.T * np.multiply(label_sv, alphas[sv_inx]) + b
        if np.sign(predict) != np.sign(y_data[i]):
            errors += 1
    print("测试数据集错误率%f" % (float(errors) / m))
