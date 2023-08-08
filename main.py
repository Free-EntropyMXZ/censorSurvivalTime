import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

N = 5000
G = 10
T = 20


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rmsve(v_standard, r_ref):
    return np.sqrt(np.mean(np.power(v_standard - r_ref, 2)))


def cutLen(t, T, G):
    y = np.zeros(G)
    delt = T / G
    for j in range(G):
        if (j + 1) * delt < t:
            y[j] = delt
        elif j * delt > t:
            y[j] = 0
        else:
            y[j] = t - j * delt
    return y


def cutOriST(ST, T, G):
    N = len(ST)
    Tau = np.zeros((N, G))
    for i in range(N):
        Tau[i] = cutLen(ST[i], T, G)
    return Tau


def generateData(N):
    X = np.zeros((N, 2))
    X0 = np.random.uniform(0, 5, N)
    X1 = np.random.uniform(0, 5, N)
    X2 = np.random.uniform(0, 5, N)
    X[:, 0] = X0
    X[:, 1] = X1
    ST = X0 + 2 * X1 + X2
    return X, ST


def censorData(X, ST):
    ST_ = ST.copy()
    flag = np.zeros(len(X))
    N = len(X)
    nonCenData = []
    for i in range(N):
        dx = (X[i, 0] + X[i, 1] - 10) / 5
        p = sigmoid(dx)
        f = np.random.binomial(1, p)
        flag[i] = f
        if f == 1:
            l = np.random.uniform(0, ST[i])
            ST_[i] = l
        else:
            nonCenData.append(ST[i])
    # print(np.mean(nonCenData))
    return ST_, flag


def doST(ST, flag, G, T):
    Tau = np.zeros((N, G))
    M = np.zeros((N, G))
    for i in range(len(ST)):
        delt = T / G
        tau = np.zeros(G)
        m = np.ones(G)
        j1 = 0
        for j in range(G):
            if ST[i] > (j + 1) * delt:
                tau[j] = delt
            elif j * delt < ST[i] < (j + 1) * delt:
                tau[j] = ST[i] - j * delt
                j1 = j
            else:
                tau[j] = 0

        if flag[i] == 1:
            m[j1:] = 0

        Tau[i] = tau
        M[i] = m
    return Tau, M


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
        return res


class DMest():
    def __init__(self, X, Tr, Y):
        indexs = np.where(Tr == 1)
        self.est = LinearRegression()
        self.est.fit(X[indexs], Y[indexs])

    def pre(self, x):
        return self.est.predict(x)


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


def MeanDM(X, M, Tau_cen, Tau):
    meanPredict = np.zeros(G)
    Tau_pre = np.zeros((N, G))
    for j in np.flip(range(G)):
        Y = Tau_cen[:, j]
        dm = DMest(X, M[:, j], Y)
        Y_ = dm.pre(X)
        Tau_pre[:, j] = Y_
        meanPredict[j] = rmsve(np.sum(Tau[:, j:], 1), np.sum(Tau_pre[:, j:], 1))
    return meanPredict


def MeanIPW(X, M, Tau_cen):
    meanPredict = np.zeros(G)
    for j in range(G):
        Y = Tau_cen[:, j]
        ipw = IPWest(X, M[:, j])
        Y_ = ipw.pre(X, M[:, j], Y)
        meanPredict[j] = np.mean(Y_)
    return meanPredict


def MeanDR(X, M, Tau_cen, Tau):
    meanPredict = np.zeros(G)
    Tau_pre = np.zeros((N, G))
    ve = np.zeros(G)
    for j in range(G):
        Y = Tau_cen[:, j]
        dr = DRest(X, M[:, j], Y)
        Y_ = dr.pre(X, M[:, j], Y)
        Tau_pre[:, j] = Y_
    for j in range(G):
        meanPredict[j] = np.sum(meanPredict[j:])
        ve[j] = rmsve(np.sum(Tau[:, j:], 1), np.sum(Tau_pre[:, j:], 1))
    return meanPredict, ve


def MeanDR_(X, M, Tau_cen, Tau):
    drMean = np.zeros(G)
    ve_DM = np.zeros(G)
    ve_DR = np.zeros(G)
    Q = np.zeros((N, G))
    for j in np.flip(range(G)):
        pre_dr = np.zeros(N)
        T = M[:, -1]
        Y = np.sum(Tau_cen[:, j:], 1)
        dm = DMest(X, T, Y)
        y = dm.pre(X)
        ve_DM[j] = rmsve(np.sum(Tau[:, j:], 1), y)
        ipw = LogisticRegression()
        ipw.fit(X, M[:, j])
        p = ipw.predict_proba(X)[:, 1]
        for i in range(N):
            pre_dr[i] = y[i]
            if M[i, j] == 1:
                rho = 1 / p[i]
                if j == G - 1:
                    pre_dr[i] += rho * (Tau_cen[i, j] - y[i])
                else:
                    pre_dr[i] += rho * (Tau_cen[i, j] + Q[i, j + 1] - y[i])
                    # pre_dr[i] += rho * (Tau_cen[i, j] + y_pre[i] - y[i])

                    # pre_dr[i] += rho * (np.sum(Tau_cen[i, j:]) - y[i])
        Q[:, j] = pre_dr
        y_pre=y
        ve_DR[j] = rmsve(np.sum(Tau[:, j:], 1), pre_dr)
        drMean[j] = np.mean(pre_dr)
    return drMean, ve_DM, ve_DR


def generateCensoredData(N):
    X, ST = generateData(N)
    ST_cen, flag = censorData(X, ST)
    Tau_cen, M = doST(ST_cen, flag, G, T)
    Tau = cutOriST(ST, T, G)
    return X, M, Tau_cen, Tau


def changeform(arr):
    a = arr.copy()
    for i in range(G):
        a[i] = np.sum(arr[i:])
    return a


# j=5
# X,ST=generateData(N)
# Tau=cutOriST(ST,T,G)
# ST_cen,flag=censorData(X,ST)
# Tau_cen,M=doST(ST_cen,flag,G,T)
# es=DMest(X,M[:,j],Tau_cen[:,j])
# y_=es.pre(X)
# print(ST[:10])
# print(M[:10,j])
# print(Tau[:10,j])
# print(y_[:10])
# print(np.mean(y_    ),np.mean(Tau[:,j]))


X, M, Tau_cen, Tau = generateCensoredData(N)
mean = np.load('mean.npy')
mean_ = np.load('mean_.npy')
# print(mean_)
# # print(mean)
# print(changeform(MeanDM(X,M,Tau_cen)))
# print(changeform(MeanIPW(X,M,Tau_cen)))
# print(changeform(MeanDR(X,M,Tau_cen)))
print(MeanDM(X, M, Tau_cen, Tau))
print(MeanDR(X, M, Tau_cen, Tau)[1])
print(MeanDR_(X, M, Tau_cen, Tau)[1])
print(MeanDR_(X, M, Tau_cen, Tau)[2])
