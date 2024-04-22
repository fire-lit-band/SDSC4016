import numpy as np
import pandas as pd
# df=pd.read_csv('./data/prob.csv')
# print(df.shape)
# from scipy.special import softmax
#
#
#
# dftrue=pd.read_csv('./data/data.csv')
# df['business_id']=dftrue['business_id']
# df['user_id']=dftrue['user_id']
# df.to_csv('prob1.csv',index=False)

df=pd.read_csv('user_list.csv')
print((df['user_id']==1).sum())