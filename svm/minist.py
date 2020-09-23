import numpy as np
from .svmMLIA import smop,kernel_trans
'手写识别'



def img2vector(filename):
    returnVec = np.zeros((1,1024))
    with open(filename) as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                returnVec[0,32*i+j] = int(line[j])
    return returnVec


def loadimg(dirname):
    from os import listdir
    labels=[]
    file_list = listdir(dirname)
    m = len(file_list)
    x_data = np.zeros((m,1024))
    for i in range(m):
        file_str = file_list[i].split('.')[0]
        label = int(file_str.split('_')[0])
        if label==9:labels.append(-1)
        else:labels.append(1)
        x_data[i,:] = img2vector('%s/%s'%(dirname,file_list[i]))
    return x_data,labels


def test_digits(kTup=('rbf',10)):
    x_data,y_data = loadimg('1trainingDigits')
    b,alphas = smop(x_data,y_data,200,0.0001,10000,kTup)
    x_data,y_data = np.mat(x_data),np.mat(y_data).transpose()
    sv_inx = np.nonzero(alphas.A > 0)[0]
    sv_s = x_data[sv_inx]
    label_sv = y_data[sv_inx]
    print('there are %d support vectors' % sv_s.shape[0])
    m, n = x_data.shape
    errors = 0
    for i in range(m):
        kernel_e_val = kernel_trans(sv_s, x_data[i, :], ('rbf', kl))
        predict = kernel_e_val.T * np.multiply(label_sv, alphas[sv_inx]) + b
        if np.sign(predict) != np.sign(y_data[i]): errors += 1
    print('训练数据集错误率%f' % (float(errors) / m))
    errors = 0
    x_test, y_test = loadimg('testDigits')
    x_test, y_test = np.mat(x_test), np.mat(y_test).transpose()
    m, n = x_test.shape
    for i in range(m):
        kernel_e_val = kernel_trans(sv_s, x_data[i, :], ('rbf', kl))
        predict = kernel_e_val.T * np.multiply(label_sv, alphas[sv_inx]) + b
        if np.sign(predict) != np.sign(y_data[i]): errors += 1
    print('测试数据集错误率%f' % (float(errors) / m))