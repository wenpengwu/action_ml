"boosting 的弱分类去权重是通过计算得到的,权重代表了分类器在上一轮迭代中的成功度，而bagging 的权重是一样的,"

"这里boosting 用的是 AdaBoost"
import numpy as np

r"""
 \xi = \frac{未正确分类数}{样本总数}
\frac{1}{2}\ln (\frac{1-\xi}{\xi})
"""


def load_simple_data():
    x_data = np.mat([[1.0, 2.1], [2.0, 1.1], [1.3, 1], [1, 1], [2.0, 1]])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return x_data, labels


def stump_classify(x_data, dimen, thresh_val, thresh_ineq):
    ret_arr = np.ones(np.shape(x_data)[0], 1)
    if thresh_ineq == "lt":
        ret_arr[x_data[:, dimen] <= thresh_val] = -1.0
    else:
        ret_arr[x_data[:, dimen] > thresh_val] = -1.0
    return ret_arr


"找到最优单层决策树"


def build_stump(x_data, labels, D):
    x_data, labels = np.mat(x_data), np.mat(labels).transpose()
    m, n = x_data.shape
    num_steps = 10.0
    best_stump = {}
    best_clas_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf
    for i in range(n):
        range_min, range_max = x_data[:, i].min(), x_data[:, i].max()
        step_size = (range_max - range_min) / num_steps
        for j in range(-1, int(num_steps) + 1):
            for inequal in ["lt", "gt"]:
                thresh_val = range_min + float(j) * step_size
                pre = stump_classify(x_data, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[pre == labels] = 0
                weighted_error = D.T * err_arr
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_clas_est = pre.copy()
                    best_stump = {"dim": i, "thresh": thresh_val, "ineq": inequal}
    return best_stump, min_error, best_clas_est


def adaboost_trainDS(x_data, labels, iter_num=40):
    week_classify_list = []
    m = np.shape(x_data)[0]
    D = np.mat(np.ones((m, 1)) / m)
    agg_class_est = np.mat(np.zeros((m, 1)))

    for i in range(iter_num):
        best_stump, error, class_est = build_stump(x_data, labels, D)
        print("D:", D.T)
        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))
        best_stump["alpha"] = alpha
        week_classify_list.append(best_stump)
        print("classEst:", class_est.T)
        expon = np.multiply(-1 * alpha * np.mat(labels).T, class_est)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        agg_class_est += alpha * class_est
        print("agg class Est", agg_class_est.T)
        agg_errors = np.multiply(
            np.sign(agg_class_est) != np.mat(labels).T, np.ones((m, 1))
        )
        rate = agg_errors.sum() / m
        print("total error: ", rate)
        if rate == 0.0:
            break

    return week_classify_list


def adaClassify(x_test, classify_list):
    x_test = np.mat(x_test)
    m = x_test.shape[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classify_list)):
        class_est = stump_classify(
            x_test,
            classify_list[i]["dim"],
            classify_list[i]["thresh"],
            classify_list[i]["ineq"],
        )
        agg_class_est += classify_list[i]["alpha"] * class_est
        print(agg_class_est)
    return np.sign(agg_class_est)
