import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt

N = 892 #number of samples
G = 1 #number of intervals
T = 17 #the total survival time for division

#random censor given the survival time of an individual, follows an flipped exponential distribution
def explen(T):
    bet = 5
    a = np.random.exponential(scale=bet)
    if a < 2 * bet:
        return T - a * T / (2 * bet)
    else:
        return explen(T)

#sigmoid function, gives the probability of censoring for a sample
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#root mean square error
def rmsve(v_standard, r_ref):
    return np.sqrt(np.mean(np.power((v_standard - r_ref), 2)))

#cut the survival time of one individual to G intervals
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

#cut the original uncensored N-samples dataset to a N*G array of G intervals
def cutOriST(ST, T, G):
    N = len(ST)
    Tau = np.zeros((N, G))
    for i in range(N):
        Tau[i] = cutLen(ST[i], T, G)
    return Tau

#generate the dataset of X and survival time
def generateData(N):
    X = np.zeros((N, 2))
    X0 = np.random.uniform(0, 5, N)
    X1 = np.random.uniform(0, 5, N)
    X2 = np.random.uniform(0, 5, N)
    X[:, 0] = X0
    X[:, 1] = X1
    ST = X0 + 2 * X1 + X2
    return X, ST

#do random censoring to the original dataset, flag=1 if we decide to censor this patient
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
            l = explen(ST[i])
            ST_[i] = l
        else:
            nonCenData.append(ST[i])
    # print(np.mean(nonCenData))
    return ST_, flag

#analyse the censored data, generate M_ij for each individual i at each interval j
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

#doubly robust estimator
class DRest():
    def __init__(self, X, Tr, Tau):
        indexs = np.where(Tr == 1)
        self.drest = LinearRegression()
        self.ipwest = LogisticRegression(max_iter=5000)
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
                res[i] = y_[i]
        return res

#direct method estimator, here we use linear regression
class DMest():
    def __init__(self, X, Tr, Y):
        indexs = np.where(Tr == 1)
        self.est = LinearRegression()
        self.est.fit(X[indexs], Y[indexs])

    def pre(self, x):
        return self.est.predict(x)

#IPW estimator
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

#DM estimator for G intervals, returning rmsve
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

#IPW estimator for G intervals, returning rmsve
def MeanIPW(X, M, Tau_cen):
    meanPredict = np.zeros(G)
    for j in range(G):
        Y = Tau_cen[:, j]
        ipw = IPWest(X, M[:, j])
        Y_ = ipw.pre(X, M[:, j], Y)
        meanPredict[j] = np.mean(Y_)
    return meanPredict

#DR estimator for G intervals, returning rmsve
def MeanDR(X, M, Tau_cen):
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
        # ve[j] = rmsve(np.sum(Tau[:, j:], 1), np.sum(Tau_pre[:, j:], 1))
    return Tau_pre

#another DR estimator for G intervals, returning rmsve of DM and DR, details can be seen in appendix
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
        y_pre = y
        ve_DR[j] = rmsve(np.sum(Tau[:, j:], 1), pre_dr)
        drMean[j] = np.mean(pre_dr)
    return ve_DM, ve_DR


def generateCensoredData(N):
    numCen = np.zeros(G)
    X, ST = generateData(N)
    Tau = cutOriST(ST, T, G)
    # ST+=np.random.normal(0,1,size=N)
    ST_cen, flag = censorData(X, ST)
    Tau_cen, M = doST(ST_cen, flag, G, T)
    for j in range(G):
        numCen[j] = N - np.sum(M[:, j])
    numCen=numCen/N
    return X, M, Tau_cen, Tau, numCen


def changeform(arr):
    a = arr.copy()
    for i in range(G):
        a[i] = np.sum(arr[i:])
    return a


def plotBar(ve_DR, ve_DM, Num_cen):
    barWidth = 0.25
    br = np.arange(G)
    xlabels = [(i + 1).__str__() for i in range(G)]
    fig, ax1 = plt.subplots()
    ax1.bar(br, ve_DR, color='r', width=barWidth, label='DR')
    ax1.bar(br + barWidth, ve_DM, color='g', width=barWidth, label='DM')
    ax1.set_ylabel('rmsve', fontweight='bold', fontsize=10)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Num_censor',fontweight='bold', fontsize=10)
    ax2.plot(br, Num_cen)

    plt.xlabel('intervals', fontweight='bold', fontsize=15)
    plt.xticks([r for r in range(G)], xlabels)
    ax1.legend()
    # ax1.ylim(0,1.5)
    plt.savefig("squares.pdf")
    plt.show()

if __name__=="__main__":
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


    # X, M, Tau_cen, Tau, numC = generateCensoredData(N)
    # mean = np.load('mean.npy')
    # mean_ = np.load('mean_.npy')
    # ve_DM = MeanDM(X, M, Tau_cen, Tau)
    # ve_DR = MeanDR(X, M, Tau_cen, Tau)

    X=np.load('X.npy')
    delt=np.load('delt.npy')
    Y=np.load('Y.npy')
    Tau,M = doST(Y,delt, G, T)
    Dr=MeanDR(X,M,Tau)
    Dr=np.sum(Dr,1)
    Drp=pd.DataFrame(Dr,columns=['pseudoST'])
    Drp.to_csv('pseudo.csv',index=False)

