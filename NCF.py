#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset


# In[2]:


NUM_USERS = 8287
NUM_ITEMS = 113613


# ### MF Model

# In[10]:


class NeuMF(torch.nn.Module):
    def __init__(self, config):
        super(NeuMF, self).__init__()
        self.config = config
        self.num_users = NUM_USERS
        self.num_items = NUM_ITEMS
        self.latent_dim_mf = config['latent_dim_mf']
        self.latent_dim_mlp = config['latent_dim_mlp']

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.Dropout(p = 0.3))

        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.logistic = torch.nn.ReLU()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = torch.squeeze(logits)
        return rating

   `


# ### Necessary Functions

# In[26]:


def like_generator(df, ratio=0.5):
    # Grouping and sorting
    grouped = df.groupby('uid').apply(lambda x: x.sort_values('rating', ascending=False))
    # Calculate count for each group
    count = np.ceil(grouped.groupby(level=0).size() * ratio).astype(int)
    # Filter data based on count
    filtered_data = grouped.groupby(level=0).apply(lambda x: x.head(count[x.name]))
    return filtered_data.reset_index(drop=True)

def _predict(uid_ori, items_ori, n, model):
    uid = torch.tensor(uid_ori).to(device)
    items = torch.tensor(items_ori).to(device)
    with torch.no_grad():
        scores = model(uid, items)
    if n > scores.shape[0]: 
        n = scores.shape[0]
    top_N_val, top_N_idx = torch.topk(scores, k=n)
    
    if n == 1:
        return [(top_N_idx.cpu().item(), top_N_val.cpu().item())]
    
    return list(zip(items_ori[top_N_idx.cpu()], top_N_val.cpu()))

def NDCG(uid, n, test_df):         # 用模型排序+真实分数计算 DCG, 重排后计算 iDCG
    # test 集中，uid 评过的 items
    test_user = test_df[test_df['uid'] == uid]
    
    # 对这些 items 做 top-k
    rating = _predict(test_user.iloc[:, 0].values, test_user.iloc[:, 1].values, n, model)
#     print(rating)
    # 排序真实评分
    irating =sorted(test_user.iloc[:, 2].values, reverse=True)
    irating = np.asarray(irating)
    
    if n > len(irating): n = len(irating) 
        
    # 取出模型排序下 merge 到的真实分数    
    rating_df = pd.DataFrame(rating, columns=['iid', 'pred_rating'])
    merged_df = pd.merge(rating_df, test_user, on='iid')
    r = np.array(merged_df['rating'])    
        
    # 求 log 分母
    log = np.log(np.arange(2, n + 2))
    
    # 求 dcg 和 idcg
    dcg = np.log(2) * np.sum((2**r[:n] - 1) / log)
    idcg = np.log(2) * np.sum((2**irating[:n] - 1) / log)
    
    return dcg / idcg

def performance(n, model, user_items, like_user_items, test_df):      # Output recall@n, precision@n, NDCG@n
    hit = 0
    n_recall = 0
    n_precision = 0
    ndcg = 0
    iid = np.arange(NUM_ITEMS)
    for i in range(NUM_USERS):
        # Items that User i tried in testing set
        unknown_items = user_items[i]
        
        # Items that User i likes testing set
        known_items = like_user_items[i]

        #目标：预测 unknown items 中的top_N，若击中test中的items，则为有效预测
        ru = _predict(len(unknown_items)*[i], unknown_items, n, model)

        hit += sum(1 for item, pui in ru if item in known_items)
        n_recall += len(known_items)
        n_precision += n
        ndcg += NDCG(i, n, test_df)

    recall = hit / (1.0 * n_recall)
    precision = hit / (1.0 * n_precision)
    ndcg /= NUM_USERS
    return recall, precision, ndcg


# ### Hyper parameters

# In[24]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 256
epoch = 100
lr = 5e-4

# 评价指标
n = 5

config = {
    'latent_dim_mf': 128,
    'latent_dim_mlp': 128,
    'layers': [256, 512, 512],
    
    # 原始 para
    # 'latent_dim_mf': 8,
    # 'latent_dim_mlp': 8,
    # 'layers': [16,32,16,8],
    
    # layers[0] is the concat of latent user vector & latent item vector
}
print(config)
print(batch_size)
print(epoch)
print(lr)

# ### Data Preparation

# In[6]:


data_dir = '../autodl-fs/SDSC4016_Project/data'
train = pd.read_csv(os.path.join(data_dir, "train.csv"))

test = pd.read_csv(os.path.join(data_dir, "test.csv"))
test_like = like_generator(test)

# 用户u对应他访问过的所有items集合
train_user_items = train.groupby('uid')['iid'].apply(lambda x: np.array(x)).to_dict()

test_user_items = test.groupby('uid')['iid'].apply(lambda x: np.array(x)).to_dict()
test_like_user_items = test_like.groupby('uid')['iid'].apply(lambda x: np.array(x)).to_dict()

# 创建训练集张量
train_data = torch.tensor(train[['uid', 'iid']].values, dtype=torch.long).to(device)
train_targets = torch.tensor(train['rating'].values, dtype=torch.float).to(device)

# 创建测试集张量
test_data = torch.tensor(test[['uid', 'iid']].values, dtype=torch.long).to(device)
test_targets = torch.tensor(test['rating'].values, dtype=torch.float).to(device)

# 使用 TensorDataset 封装数据
train_dataset = TensorDataset(train_data, train_targets)
test_dataset = TensorDataset(test_data, test_targets)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ### Training

# In[27]:


# NCF model
model = NeuMF(config).to(device)
model.device = device

# Mean Sqaured Error
criterion = nn.MSELoss()

# Adam optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)     # 主模型优化器

for x in range(epoch):
    # Training
    model.train()
    train_loss = 0

    for batch, rating in train_loader:
        uids = batch[:, 0]
        iids = batch[:, 1]
        
        pred_rating = model(uids, iids)
        loss = criterion(pred_rating, rating)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f'Train | [{x+1}/{epoch}], Loss: {train_loss/len(train_loader)}')
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch, rating in test_loader:
            uids = batch[:, 0]
            iids = batch[:, 1]
            
            pred_rating = model(uids, iids)
            loss = criterion(pred_rating, rating)
            
            val_loss += loss.item()
    
    rec, pre, ndcg = performance(n, model, test_user_items, test_like_user_items, test)
    
    print(f'Valid | [{x+1}/{epoch}], Loss: {val_loss/len(test_loader)}, Pre@{n}: {pre}, Rec@{n}: {rec}, NDCG@{n}: {ndcg}')

    print('---------------------------------------------------')


# In[ ]:




