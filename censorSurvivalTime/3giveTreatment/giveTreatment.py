import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

tf2 = [['age_dx'],
       ['PromiseSubtype', 'esmo2020_mol', 'lvi', 'MyoInvasion', 'hist_surgery_dx',
        'grade_surgery_b12v3', 'stage', 'nodes', 'ace_27_any']]
tf1 = [['age_dx', 'bmi', 'Ca125'],
       ['PromiseSubtype_dx', 'grade_dx_bx_b12v3', 'hist_dx', 'ace_27_any', 'Extrauterine_Disease']]
blp2 = [[], ['esmo2020_mol', 'lvi', 'MyoInvasion', 'hist_surgery_dx',
             'grade_surgery_b12v3', 'stage_surgery_b1v2v34', 'nodes', 'PromiseSubtype']]
blp1 = [['age_dx', 'bmi'],
        ['PromiseSubtype_dx', 'grade_dx_bx_b12v3', 'hist_dx', 'ace_27_any', 'Extrauterine_Disease']]

X = pd.read_csv('out.csv')
X_tf2 = pd.concat([X[tf2[0]], pd.get_dummies(X[tf2[1]])], axis=1)
X_blp2 = pd.get_dummies(X[blp2[1]])

N = len(X)

T2 = np.zeros((N, 3))
T2_dum = pd.get_dummies(X['treat2']);
T2_dum = np.array(T2_dum)
T2[:, 0] = T2_dum[:, 1];
T2[:, 1] = T2_dum[:, 0];
T2[:, 2] = T2_dum[:, 2]

ps2 = pd.read_csv('ps2.csv')
ps2 = np.array(ps2)


def weig():
    wei = np.zeros(N)
    for i in range(N):
        index = np.where(T2[i])[0][0]
        wei[i] = 1.0 / ps2[i, index]
    return wei


def Xfit2(tr):
    Ltf = len(np.array(X_tf2)[0])
    Lblp = len(np.array(X_blp2)[0])
    X_fit = np.zeros((N, Ltf + 2 * Lblp))
    X_fit[:, :Ltf] = np.array(X_tf2)
    for i in range(N):
        if tr[i, 1] == 1:
            X_fit[i, Ltf:Ltf + Lblp] = np.array(X_blp2)[i]
        if tr[i, 2] == 1:
            X_fit[i, Ltf + Lblp:Ltf + 2 * Lblp] = np.array(X_blp2)[i]
    return X_fit


def Xpre2(num):
    Ltf = len(np.array(X_tf2)[0])
    Lblp = len(np.array(X_blp2)[0])
    x = np.zeros((N, Ltf + 2 * Lblp))
    if num == 0:
        x[:, :Ltf] = np.array(X_tf2)
    if num == 1:
        x[:, Ltf:Ltf + Lblp] = np.array(X_blp2)
    if num == 2:
        x[:, Ltf + Lblp:Ltf + 2 * Lblp] = np.array(X_blp2)
    return x


def Fit():
    X_fit = Xfit2(T2)
    Y = np.array(pd.read_csv('pseudo.csv'))
    Y=Y[:,0]
    lg = LinearRegression()
    wei = weig()
    lg.fit(X_fit, Y)
    y=lg.predict(X_fit)
    pd.DataFrame(y).to_csv('Y_pre.csv')
    return lg


def out():
    lg=Fit()
    X0=Xpre2(0)
    Y0=lg.predict(X0)
    X1=Xpre2(1)
    Y1=lg.predict(X1)-lg.intercept_
    X2=Xpre2(2)
    Y2=lg.predict(X2)-lg.intercept_
    res=np.stack((Y0,Y1,Y2),axis=1)
    df=pd.DataFrame(res,columns=['tf','blp1','blp2'])
    df.to_csv('Blip.csv',index=False)
def Ind(n):
    return np.where(T2[:,n]==1)

def anothertry():
    wei = weig()
    Y = np.array(pd.read_csv('pseudo.csv'))
    Y = Y[:, 0]
    lg0=LinearRegression().fit(np.array(X_tf2)[Ind(0)],Y[Ind(0)],wei[Ind(0)])
    lg1=LinearRegression().fit(np.array(X_blp2)[Ind(1)],Y[Ind(1)],wei[Ind(1)])
    lg2 = LinearRegression().fit(np.array(X_blp2)[Ind(2)], Y[Ind(2)], wei[Ind(2)])
    res = np.stack((lg0.predict(X_tf2), lg1.predict(X_blp2), lg2.predict(X_blp2)), axis=1)
    df = pd.DataFrame(res, columns=['tf', 'blp1', 'blp2'])
    df.to_csv('try.csv', index=False)
anothertry()