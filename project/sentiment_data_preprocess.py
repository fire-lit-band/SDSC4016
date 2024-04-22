import pandas as pd

# 1.read the origin data
df= pd.read_csv('./data/data.csv')
print(df.shape)
df=df[['stars','text']]
print(df.shape)

# 2.extract the comment data
df.to_csv('./data/origin_sentiment_data.csv',index=False)

# 3.read the comment data
# df=pd.read_csv('./data/origin_sentiment_data.csv')

# 4.split into train and test data
from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.2)
# train.to_csv('./data/train_sentiment_data.csv',index=True)
# test.to_csv('./data/test_sentiment_data.csv',index=True)

# train=pd.read_csv('./data/train_sentiment_data.csv')
#
# text=[i.split(' ') for i in train['text']]
# print(text[0])
# text_len=[len(i.split(' ')) for i in train['text']]
# import numpy as np
# text_len=np.array(text_len)
# print(len(text_len)/100)
# print(len(text_len[text_len>500])/len(text_len))
# import matplotlib.pyplot as plt
# plt.hist(text_len,bins=100)
# plt.show()




