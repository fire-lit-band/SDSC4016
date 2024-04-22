from transformers import BertTokenizer, TFBertForSequenceClassification

# needed for fine tune the model
import tensorflow as tf
from sklearn.model_selection import train_test_split

# needed for data pre-processing
import pandas as pd
import numpy as np
from os.path import basename


def load_dataset(list_files):
    data_list = []
    for file in list_files:
        lines = open(file, 'r').readlines()
        data_list += [[line.strip(), int(basename(file)[-1])] for line in lines]
    data = np.array(data_list)
    np.random.shuffle(data)
    return data


def split_dataset(data):
    return train_test_split(
        data[:, 0],
        data[:, 1],
        test_size=0.2,
        random_state=123
    )


def create_tf_dataset(X, y, tokenizer, max_length=128):
    features = []
    for text, label in zip(X, y):
        input_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding='max_length',
            truncation=True
        )

        features.append(
            (
                {
                    "input_ids": input_dict["input_ids"],
                    "attention_mask": input_dict['attention_mask'],
                    "token_type_ids": input_dict["token_type_ids"],
                },
                label,
            )
        )

    def gen():
        for f in features:
            yield f

    return tf.data.Dataset.from_generator(
        gen,
        ({
             "input_ids": tf.int32,
             "attention_mask": tf.int32,
             "token_type_ids": tf.int32
         },
         tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
# load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

data_train = load_dataset(
    ['/kaggle/input/yelp-dataset/sentiment.train.0',
     '/kaggle/input/yelp-dataset/sentiment.train.1']
)
data_test = load_dataset(
    ['/kaggle/input/yelp-dataset/sentiment.test.0',
     '/kaggle/input/yelp-dataset/sentiment.test.1']
)
print("training data size: "+str(len(data_train)))
print("testing data size: "+str(len(data_test)))
print("train datset head:")
print(data_train[:7])
print("test datset head:")
print(data_test[:7])