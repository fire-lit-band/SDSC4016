import os
import shutil
import tarfile
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd

import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
model = TFBertForSequenceClassification.from_pretrained('./bert-base-uncased', num_labels=2)
# load the data

df_train = pd.read_csv('/root/autodl-fs/train_sentiment_data.csv')

Reviews= df_train['text']
Target = df_train['stars']
# 让>=4的评分为1，<4的评分为0
Target[Target<=2]=0
Target[Target>=3]=1
# Target[Target>=4]=2
df_test= pd.read_csv('/root/autodl-fs/test_sentiment_data.csv')
df_test=df_test.sample(frac=0.001)
x_test= df_test['text']
y_test= df_test['stars']
y_test[y_test<=2]=0
y_test[y_test>=3]=1
# y_test[y_test>=4]=2
print('Data loaded')

print('Tokenizer loaded')
Reviews, x_val, Target, y_val = train_test_split(Reviews,
                                                    Target,
                                                    test_size=0.1,
                                                    stratify = Target)

print('Train test split done')
max_len = 128
print('Max length set')


# Tokenize and encode the sentences
X_train_encoded = tokenizer.batch_encode_plus(Reviews.tolist(),
                                              padding=True,
                                              truncation=True,
                                              max_length=max_len,return_tensors='tf')
print('train_Tokenization done')
X_val_encoded = tokenizer.batch_encode_plus(x_val.tolist(),
                                            padding=True,
                                            truncation=True,
                                            max_length=max_len,
                                            return_tensors='tf')
print('val_Tokenization done')
X_test_encoded = tokenizer.batch_encode_plus(x_test.tolist(),
                                             padding=True,
                                             truncation=True,
                                             max_length=max_len,
                                             return_tensors='tf')
print('Tokenization done')

print('Model loaded')

# train the model
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import datetime
# 定义 ModelCheckpoint 回调
time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=f'/root/tf-logs/bert-sentiment-analysis{time}', histogram_freq=1, update_freq='epoch')

checkpoint_loss = ModelCheckpoint('best_model_loss', monitor='val_loss', mode='min', save_best_only=True, save_format='tf')
checkpoint_acc = ModelCheckpoint('best_model_accuracy', monitor='val_accuracy', mode='max', save_best_only=True, save_format='tf')

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5,epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
print('Model compiled')

with tf.device('/GPU:0'):
    history = model.fit(
        [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
        Target,
        validation_data=(
          [X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']],y_val),
        batch_size=32,
        epochs=300,
        callbacks=[checkpoint_loss,checkpoint_acc,tensorboard_callback ] ,steps_per_epoch=1 ,validation_batch_size=32,shuffle=True# 在 fit 函数中添加 callbacks 参数
    )

test_loss, test_accuracy = model.evaluate(
    [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
    y_test
)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

path = 'path-to-save'
# Save tokenizer
tokenizer.save_pretrained(path + '/Tokenizer')

# Save model
model.save_pretrained(path + '/Model')