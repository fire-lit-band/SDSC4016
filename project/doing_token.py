import os
import shutil
import tarfile
from transformers import BertTokenizer, TFBertForSequenceClassification,BertForSequenceClassification
import pandas as pd

# import re
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# # from sklearn.metrics import classification_report
# tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
# # load the data
#
# df_train = pd.read_csv('/root/autodl-fs/train_sentiment_data.csv')
# Reviews= df_train['text']
# Target = df_train['stars']
# # 让>=4的评分为1，<4的评分为0
#
# # Target[Target>=4]=2
# df_test= pd.read_csv('/root/autodl-fs/test_sentiment_data.csv')
# x_test= df_test['text']
# y_test= df_test['stars']
# # y_test[y_test>=4]=2
# print('Data loaded')
#
# print('Tokenizer loaded')
# Reviews, x_val, Target, y_val = train_test_split(Reviews,
#                                                     Target,
#                                                     test_size=0.1,
#                                                     stratify = Target)
# print('Train test split done')
# max_len = 128
# print('Max length set')
# Target.to_csv('./data/Target.csv')
# y_val.to_csv('./data/y_val.csv')
#
#
# # Tokenize and encode the sentences
# X_train_encoded = tokenizer.batch_encode_plus(Reviews.tolist(),
#                                               padding=True,
#                                               truncation=True,
#                                               max_length=max_len,return_tensors='tf')
# import pickle
#
# with open('X_train_encoded.pkl', 'wb') as f:
#     pickle.dump(X_train_encoded, f)
#
# print('train_Tokenization done')
# X_val_encoded = tokenizer.batch_encode_plus(x_val.tolist(),
#                                             padding=True,
#                                             truncation=True,
#                                             max_length=max_len,
#                                             return_tensors='tf')
# with open('X_train_encoded.pkl', 'wb') as f:
#     pickle.dump(X_val_encoded, f)
# print('val_Tokenization done')
# X_test_encoded = tokenizer.batch_encode_plus(x_test.tolist(),
#                                              padding=True,
#                                              truncation=True,
#                                              max_length=max_len,
#                                              return_tensors='tf')
# print('Tokenization done')
# with open('X_train_encoded.pkl', 'wb') as f:
#     pickle.dump(X_test_encoded, f)
# print('Model loaded')

from transformers import BertModel, TFBertModel

# 加载 PyTorch 模型
pytorch_model = BertForSequenceClassification.from_pretrained('./test', from_tf=False)

# 将 PyTorch 模型转换为 TensorFlow 模型
pytorch_model.save_pretrained('./', save_format='tf')

# 加载 TensorFlow 模型
tensorflow_model = TFBertForSequenceClassification.from_pretrained('./', from_pt=False)


