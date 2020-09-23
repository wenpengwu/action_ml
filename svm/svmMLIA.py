import  numpy as np

'投影距离'


'''

W^Tx+b  W的转置乘以X + b

拉格朗日数乘法
'''
def load_dataset(filename):
    x_data ,y_data=[],[]
    with open(filename) as f:
        for line in f.readlines():
            a,b,c =line.strip().split('\t')
            x_data.append([float(a),float(b)])
            y_data.append(float(c))
    return x_data,y_data

def select_jrand(i,m):
    j=i
    while j==i:
        j=int(np.random.uniform(0,m))
    return j

def clip_alpha(aj,H,L):
    if aj>H:aj=H
    if L>aj:aj=L
    return aj


'简化版SMO'

def smo_simple(x_data,y_data,C,toler,iter_num):
    x_data = np.mat(x_data)
    y_data = np.mat(y_data).transpose()
    b=0
    m,n=x_data.shape
    alphas = np.mat(np.zeros((m,1)))
    iter =0
    while iter<iter_num:
        alpha_pairs_changed = 0
        for i in range(m):
            fxi = float(np.multiply(alphas,y_data).T*(x_data*x_data[i,:].T))+b
            ei = fxi - float(y_data[i])
            if (y_data[i]*ei<-toler and alphas[i]<C ) or (y_data[i]*ei>toler and alphas[i]>0):
                j = select_jrand(i,m)
                fxj = float(np.multiply(alphas,y_data).T*(x_data*x_data[j,:].T))+b
                ej = fxj - float(y_data[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if y_data[i]!=y_data[j]:
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L==H:
                    print('L==H')
                    continue

                eta = 2.*x_data[i,:]*x_data[j,:].T - x_data[i,:]*x_data[i,:].T - x_data[j,:]*x_data[j,:].T
                if eta>=0:
                    print('eta>=0')
                    continue
                alphas[j]-=y_data[j]*(ei-ej)/eta
                alphas[j] =clip_alpha(alphas[j],H,L)
                if abs(alphas[j]-alphaJold)<.00001:
                    print('j not moving enough')
                    continue
                alphas[i]+=y_data[j]*y_data[i]*(alphaJold-alphas[j])
                b1 = b-ei - y_data[i]*(alphas[i]-alphaIold)*x_data[i,:]*x_data[i,:].T - y_data[j]*(alphas[j]-alphaJold)*x_data[i,:]*x_data[j,:].T
                b2 = b - ej - y_data[i]*(alphas[i]-alphaIold)*x_data[i,:]*x_data[j,:].T - y_data[j]*(alphas[j]-alphaJold)*x_data[i,:]*x_data[j,:].T
                if alphas[i]>0 and C>alphas[i]:b=b1
                elif alphas[j]>0 and C>alphas[j]:b=b2
                else:b=(b1+b2)/2.
                alpha_pairs_changed+=1
                print('iter :%d i:%d. pairs change %d'%(iter,i,alpha_pairs_changed))
        if alpha_pairs_changed==0:iter+=1
        else:iter=0
        print('tier number:%d'%iter)
    return b,alphas

class OptStruct:

    def __init__(self,input_mat,labels,C,toler):
            self.X = input_mat
            self.labels = labels
            self.C=C
            self.tol = toler
            self.m = input_mat.shape[0]
            self.alphas = np.mat(np.zeros((self.m,1)))
            self.b = 0
            self.e_cache = np.mat(np.zeros((self.m,2)))

def cal_ek(os,k):
    '''

    Parameters
    ----------
    os:OptStruct
    k:int

    Returns ek:float
    -------

    '''
    fxk = float(np.multiply(os.alphas,os.labels).T *(os.X*os.X[k,:].T))+os.b
    ek = fxk - float(os.labels[k])
    return ek


def select_j(i,os,ei):
    '''

    Parameters
    ----------
    i
    os:OptStruct
    ei

    Returns
    -------

    '''
    max_k,max_detal_e,ej=-1,0,0
    os.e_cache[i] = [1,ei]
    valid_e_cache_list = np.nonzero(os.e_cache[:,0].A)[0]
    if len(valid_e_cache_list)>1:
        for k in valid_e_cache_list:
            if k==i:continue
            ek =cal_ek(os,k)
            delat_e = abs(ei-ek)
            if delat_e>max_detal_e:
                max_k=k
                max_detal_e=delat_e
                ej = ek
        return max_k,ej
    else:
        j = select_jrand(i,os.m)
        ej = cal_ek(os,j)
        return j,ej

def update_ek(os,k):
    '''

    Parameters
    ----------
    os:OptStruct
    k

    Returns
    -------

    '''
    ek = cal_ek(os,k)
    os.e_cache[k] = [1,ek]


