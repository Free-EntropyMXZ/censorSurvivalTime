import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
N=1000
def p(x):
    return 1/(1+np.exp(-(0.1*x+0.3)))
def sample(x,y,y_,n):
    N=len(x)
    indexs=np.random.randint(0,N,n)
    for i in indexs:
        print(x[i],y[i],y_[i])

x=np.random.uniform(-10,10,N)
y_=1/(1+np.exp(-(0.1*x+0.3)))
y=[]
for i in x:
    y.append(np.random.binomial(1,p(i)))
x=x.reshape((N,1))
LogModel=LogisticRegression()
LogModel.fit(x,y)

x=np.random.uniform(-10,10,5).reshape((5,1))
y_=1/(1+np.exp(-(0.1*x+0.3)))
y_pre=LogModel.predict_proba(x)
print(y_pre)