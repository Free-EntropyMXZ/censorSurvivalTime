import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt

N = 8
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear2(x, y,t):
    x_nc,y_nc=pickupDa(1,x,y,t)
    x_c,y_c=pickupDa(0,x,y,t)
    x_b,par=para(x,t,y)

    x_all,parall=para(x,np.ones(len(x)),y)
    y_all=np.dot(x_all,parall)

    y_hat = np.dot(x_b,par)

    plt.scatter(x_c,y_c,c='r',alpha=0.2)
    plt.scatter(x_nc, y_nc, c='r')
    plt.plot(x,y_all,'k',label='gt')
    plt.plot(x_nc, y_hat, 'b',label='fs1')
    plt.legend()
    plt.savefig("lr.pdf")

    plt.show()

    return par
def linear2_(x, y,t,par_):
    x_nc,y_nc=pickupDa(1,x,y,t)
    x_c,y_c=pickupDa(0,x,y,t)
    x_b,par=para(x,t,y)

    y_hat = np.dot(x_b,par)
    y_hat_=np.dot(x_b,par_)

    x_all, parall = para(x, np.ones(len(x)), y)
    y_all = np.dot(x_all, parall)
    plt.scatter(x_c,y_c,c='r',alpha=0.2)
    plt.scatter(x_nc, y_nc, c='r')
    plt.plot(x,y_all,'k',label='gt')
    plt.plot(x_nc, y_hat, 'b',label='os')
    plt.plot(x_nc, y_hat_, 'g',label='fs')
    plt.legend()
    plt.savefig("fullComp.pdf")
    plt.show()

    return par


def generateData(N):
    X = np.zeros((2 * N, 1))
    Y = np.zeros(2 * N)
    sigm = 1
    X0 = np.random.normal(-1, sigm, N)
    X1 = np.random.normal(1, sigm, N)
    X[:N, 0] = X0
    X[N:, 0] = X1
    Y[:N] += np.random.normal(0, sigm / 5, N) + 1
    Y[N:] += np.random.normal(0, sigm / 5, N) + 4

    return X, Y


def censor(X, Y):
    T = np.zeros(2 * N)
    T[:N] = 1 - np.random.binomial(1, 0.1, N)
    T[N:] = 1 - np.random.binomial(1, 0.9, N)
    return T


def pickupDa(t, X, Y, T):
    index0 = np.where(T == 0)
    index1 = np.where(T == 1)
    if t == 0:
        return X[index0], Y[index0]
    else:
        return X[index1], Y[index1]

def para(x_,t,y_):
    x,y=pickupDa(1,x_,y_,t)
    n = len(x)
    x_b = np.zeros((n, 2))
    x_b[:, 0] = 1
    for i in range(n):
        x_b[i, 1] = x[i]
    param = np.dot(np.dot(np.linalg.inv(np.dot(x_b.T, x_b)), x_b.T), np.mat(y).T)
    return x_b,param
def dm(x,t,y):
    dme=DMest(x,t,y)
    return dme.pre(x,t,y)
def dr(x,t,y):
    dre=DRest(x,t,y)
    return dre.pre(x,t,y)
def ipw(x,t,y):
    ipwe=IPWest(x,t)
    return ipwe.pre(x,t,y)

class IPWest():
    def __init__(self, X, Tr):
        self.est = LogisticRegression()
        self.est.fit(X, Tr)

    def pre(self, x, tr, y):
        total = 0
        p = self.est.predict_proba(x)[:, 1]
        for i in range(len(tr)):
            if tr[i] == 1:
                total += y[i] / p[i]
        total = total / len(tr)
        return total


class DMest():
    def __init__(self, X, Tr, Y):
        indexs = np.where(Tr == 1)
        self.est = LinearRegression()
        self.est.fit(X[indexs], Y[indexs])

    def pre(self, x,T,y):
        return np.mean(self.est.predict(x))


class DRest():
    def __init__(self, X, Tr, Tau):
        indexs = np.where(Tr == 1)
        self.drest = LinearRegression()
        self.ipwest = LogisticRegression()
        self.drest.fit(X[indexs], Tau[indexs])
        self.ipwest.fit(X, Tr)

    def pre(self, x, tr, y):
        res = np.zeros(len(x))
        p = self.ipwest.predict_proba(x)[:, 1]
        y_ = self.drest.predict(x)
        for i in range(len(tr)):
            res[i] = y_[i]
            if tr[i] == 1:
                rho = 1 / p[i]
                res[i] += rho * (y[i] - y_[i])
                # res[i] = y[i]
        return np.mean(res)

def do1exp():
    x, y = generateData(N)
    T = censor(x, y)

    y0 = y.copy()
    y0[N:] = 2
    y1=y-y0
    t0 = np.zeros(N * 2)
    t0[:N] = T[:N]
    t0[N:] = 1

    os=dr(x,T,y)
    fs=dr(x,t0,y0)+dr(x,T,y1)
    _,pa0=para(x,t0,y0)
    _,pa1=para(x,T,y1)
    pa=pa0+pa1

    # print(ipw(x,T,y1))
    # print(os,fs)
    # print(dr(x, t0, y0) + dr(x, T, y1))
    # print(dr(x, T, y))
    # linear2_(x,y,T,pa)
    # linear2(x,y0,t0)
    return os,fs

epi=1000
val=np.zeros((epi,2))
for i in range(epi):
    val[i,0],val[i,1]=do1exp()
print(np.mean(val[:,0]),np.mean(val[:,1]))

# do1exp()