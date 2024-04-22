import os
import shutil
import tarfile
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd

import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# load the data
df_train = pd.read_csv('/root/autodl-fs/train_sentiment_data.csv')
df_train=df_train.sample(frac=0.1)
Reviews= df_train['text']
Target = df_train['stars']
Target=Target-1
df_test= pd.read_csv('/root/autodl-fs/test_sentiment_data.csv')
df_test=df_test.sample(frac=0.001)
x_test= df_test['text']
y_test= df_test['stars']
y_test=y_test-1
print('Data loaded')
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
print('Tokenizer loaded')
# Initialize the tokenizer

max_len = 128


def encode_examples(reviews, targets):
    # Prepare our text into dictionary format for BERT
    encoded = tokenizer.batch_encode_plus(reviews,
                                          padding='max_length',
                                          truncation=True,
                                          max_length=max_len,
                                          return_tensors='tf')

    # Split dictionary keys into separate tensors
    input_ids = encoded['input_ids']
    token_type_ids = encoded['token_type_ids']
    attention_mask = encoded['attention_mask']

    return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}, targets

import tensorflow as tf
def data_generator(reviews, targets, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((reviews, targets))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    dataset = dataset.map(encode_examples).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset


# Split the data
Reviews, x_val, Target, y_val = train_test_split(Reviews, Target, test_size=0.2, stratify=Target)
Target=Target-1
y_val=y_val-1
# Create train and validation datasets
batch_size = 32
train_dataset = data_generator(Reviews, Target, batch_size)
val_dataset = data_generator(x_val, y_val, batch_size)
model=TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=5)
import datetime

# Define the TensorBoard callback
log_dir = f'/root/tf-logs/bert-sentiment-analysis{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

# Define the ModelCheckpoint callbacks
checkpoint_loss = tf.keras.callbacks.ModelCheckpoint('best_model_loss.h5', monitor='val_loss', mode='min', save_best_only=True)
checkpoint_acc = tf.keras.callbacks.ModelCheckpoint('best_model_accuracy.h5', monitor='val_accuracy', mode='max', save_best_only=True)

# Add the callbacks to the model's fit method
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, callbacks=[checkpoint_loss, checkpoint_acc, tensorboard_callback],steps_per_epoch=1)




test_dataset = data_generator(x_test, y_test, batch_size)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

path = 'path-to-save'
# Save tokenizer
tokenizer.save_pretrained(path + '/Tokenizer')

# Save model
model.save_pretrained(path + '/Model')