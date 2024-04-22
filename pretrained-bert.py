import os
import shutil
import tarfile
from transformers import BertTokenizer, TFBertForSequenceClassification,BertForSequenceClassification
import pandas as pd

import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('./pretrained', from_tf=False)
# load the data




print('Data loaded')
import pickle
import torch


Target = pd.read_csv('Target.csv')
Target=Target['stars']
Target[Target<=2]=0
Target[Target>2]=1
y_val = pd.read_csv('y_val.csv')
y_val=y_val['stars']
y_val[y_val<=2]=0
y_val[y_val>2]=1
y_test= pd.read_csv('y_test.csv')
y_test=y_test['stars']
y_test[y_test<=2]=0
y_test[y_test>2]=1
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 加载数据
X_train_encoded = torch.load('X_train_encoded.pt')
X_val_encoded = torch.load('X_val_encoded.pt')
X_test_encoded = torch.load('X_test_encoded.pt')



# 创建数据集
train_dataset = MyDataset(X_train_encoded, Target)
val_dataset = MyDataset(X_val_encoded, y_val)
test_dataset = MyDataset(X_test_encoded, y_test)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print('Model loaded')

# train the model
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import nn
from sklearn.metrics import accuracy_score
import datetime
# 定义 TensorBoard 回调
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=f'/root/tf-logs/bert-sentiment-analysis{time}')

# 定义优化器、损失函数和度量
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-08)
loss_fn = nn.CrossEntropyLoss()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)
validation_steps=1
# 训练模型
model.train()
for epoch in range(20):  # 这里只是一个示例，你可能需要根据你的训练集调整 epoch 的数量
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids, token_type_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # 记录训练损失
        writer.add_scalar('Train/loss', loss.item(), step)

        # 每一定步数，进行一次验证并记录验证损失和准确率
        if step % validation_steps == 0:
            model.eval()
            val_loss = 0
            val_accuracy = 0
            with torch.no_grad():
                for val_batch in validation_dataloader:
                    input_ids, token_type_ids, attention_mask, labels = tuple(t.to(device) for t in val_batch)
                    outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels)
                    val_loss += outputs.loss.item()
                    preds = torch.argmax(outputs.logits, dim=1)
                    val_accuracy += accuracy_score(labels.cpu(), preds.cpu())
            val_loss /= len(validation_dataloader)
            val_accuracy /= len(validation_dataloader)

            # 记录验证损失和准确率
            writer.add_scalar('Validation/loss', val_loss, step)
            writer.add_scalar('Validation/accuracy', val_accuracy, step)

            model.train()

# 测试模型
model.eval()
test_loss = 0
test_accuracy = 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, token_type_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        test_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        test_accuracy += accuracy_score(labels.cpu(), preds.cpu())
test_loss /= len(test_dataloader)
test_accuracy /= len(test_dataloader)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

path = 'path-to-save'
# Save tokenizer
tokenizer.save_pretrained(path + '/Tokenizer')

# Save model
model.save_pretrained(path + '/Model')
