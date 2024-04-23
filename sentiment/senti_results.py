import os
import shutil
import tarfile
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd

import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load the data

df_test= pd.read_csv('/root/autodl-fs/origin_sentiment_data.csv')
x_test= df_test['text']
# y_test= df_test['stars']
# y_test=y_test-1
# print('Data loaded')
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
# print('Tokenizer loaded')
#
#
# print('val_Tokenization done')
max_len = 128
X_test_encoded = tokenizer.batch_encode_plus(x_test.tolist(),
                                             padding=True,
                                             truncation=True,
                                             max_length=max_len,
                                             return_tensors='tf')
# print('Tokenization done')
import pickle
# with open('/root/autodl-fs/origin_encoded1.pkl', 'wb') as f:
#     pickle.dump(X_test_encoded, f)

with open('/root/autodl-fs/origin_encoded.pkl', 'rb') as f:
    X_test_encoded = pickle.load(f)


model = TFBertForSequenceClassification.from_pretrained('/root/project/path-to-save/Model1', num_labels=2)
print('Model loaded')

# train the model
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import datetime
# 定义 ModelCheckpoint 回调

import pickle



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_loss= model.predict(
    [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
    batch_size=32)
# print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
#
#
#
# # 输出是一个元组，第一个元素是logits
df = pd.DataFrame(test_loss.logits)
df.to_csv('logits1.csv', index=False)
import numpy as np
df=pd.read_csv('logits.csv')
print(df.shape)
from scipy.special import softmax

probabilities = softmax(df.values, axis=1)
probabilities=probabilities[:,1]
print(probabilities)
#
# df['label']=df.idxmax(axis=1)
# df['label']=df['label'].astype(int)
# df['prob']=probabilities
dftrue=pd.read_csv('/root/autodl-fs/data.csv')
# print(dftrue.shape)
#
#
#
# df['iid']=dftrue['business_id']
# df['uid']=dftrue['user_id']
# df.to_csv('prob.csv',index=False)
df=pd.read_csv('prob.csv')
counts = dftrue['stars'].value_counts()
print(counts)
dftrue[dftrue['stars']<=2]=0
dftrue[dftrue['stars']>2]=1
from sklearn.metrics import classification_report
print(classification_report(dftrue['stars'],df['label']))




