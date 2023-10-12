import pandas as pd
import numpy as np
categ=['ace_27_any','PromiseSubtype_dx','esmo2020_mol','stage','stage_surgery_b1v2v34','grade_dx_bx_b12v3','hist_dx','Extrauterine_Disease','treat2','esmo2020',
       'grade_surgery_b12v3','lvi','MyoInvasion','hist_surgery_dx','nodes','PromiseSubtype',
       'ASAcomorbidity','gravidity','parity','surg_type','TamoxifenUse','MenopauseStatus']
drop2X=['rownames','time2','delta','time']
data=pd.read_csv('out.csv')
# data=data.drop(labels='rownames',axis=1)
data_with_categ = pd.concat([
    data.drop(columns=categ), # dataset without the categorical features
    pd.get_dummies(data[categ], columns=categ, drop_first=False)# categorical features converted to dummies
], axis=1)
data_with_categ.to_csv('output.csv',index=False)

data=pd.read_csv('output.csv')
dataX=data.drop(columns=drop2X)
dataX.to_csv('X.csv',index=False)
X=np.array(dataX)

np.save('X.npy',X)
Y=data['time2']
Y=np.array(Y)
np.save('Y.npy',Y)
delt=data['delta']
delt=np.array(delt)
np.save('delt.npy',delt)