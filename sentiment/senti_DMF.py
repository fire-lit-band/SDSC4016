import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')


NUM_USERS = 8287
NUM_ITEMS = 113613


# Model Defination

class DMF(nn.Module):
    def __init__(self, num_users, num_items, layers, train_data, train_targets):
        super(DMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = layers[0]
        self.layers = layers

        self.user_item_matrix = torch.sparse_coo_tensor(train_data.t(), train_targets,
                                                        torch.Size((self.num_users, self.num_items))).to_dense()

        self.linear_user_1 = nn.Linear(in_features=self.num_items, out_features=self.latent_dim)
        self.linear_user_1.weight.detach().normal_(0, 0.01)
        self.linear_item_1 = nn.Linear(in_features=self.num_users, out_features=self.latent_dim)
        self.linear_item_1.weight.detach().normal_(0, 0.01)

        self.user_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.user_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))
            self.user_fc_layers.append(nn.Dropout(p = 0.3))

        self.item_fc_layers = nn.ModuleList()
        for idx in range(1, len(self.layers)):
            self.item_fc_layers.append(nn.Linear(in_features=self.layers[idx - 1], out_features=self.layers[idx]))
            self.item_fc_layers.append(nn.Dropout(p = 0.3))

    def forward(self, user_indices, item_indices):

        user = self.user_item_matrix[user_indices]
        item = self.user_item_matrix[:, item_indices].t()

        user = self.linear_user_1(user)
        item = self.linear_item_1(item)

        for idx in range(len(self.layers) - 1):
            user = F.relu(user)
            user = self.user_fc_layers[idx](user)

        for idx in range(len(self.layers) - 1):
            item = F.relu(item)
            item = self.item_fc_layers[idx](item)

        vector = torch.sum(user * item, dim=1)

        return vector
    

# Necessary Funcs

def like_generator(df, ratio=0.5):
    # Grouping and sorting
    grouped = df.groupby('uid').apply(lambda x: x.sort_values('rating', ascending=False))
    # Calculate count for each group
    count = np.ceil(grouped.groupby(level=0).size() * ratio).astype(int)
    # Filter data based on count
    filtered_data = grouped.groupby(level=0).apply(lambda x: x.head(count[x.name]))
    return filtered_data.reset_index(drop=True)

def _predict(uid, items, n, model):
    with torch.no_grad():
        scores = model(uid, items)
    if n > scores.shape[0]: 
        n = scores.shape[0]
    top_N_val, top_N_idx = torch.topk(scores, k=n)
    
    if n == 1:
        return [(top_N_idx.cpu().item(), top_N_val.cpu().item())]
    
    return list(zip(items[top_N_idx.cpu()], top_N_val.cpu()))

def NDCG(uid, n, test_df):         # 用模型排序+真实分数计算 DCG, 重排后计算 iDCG
    # test 集中，uid 评过的 items
    test_user = test_df[test_df.iloc[:, 0] == uid]
    
    # 对这些 items 做 top-k
    rating = _predict(uid, test_user.iloc[:, 1].values, n, model)
    
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
        ru = _predict(i, unknown_items, n, model)

        hit += sum(1 for item, pui in ru if item in known_items)
        n_recall += len(known_items)
        n_precision += n
        ndcg += NDCG(i, n, test_df)

    recall = hit / (1.0 * n_recall)
    precision = hit / (1.0 * n_precision)
    ndcg /= NUM_USERS
    return recall, precision, ndcg


# Hyper paras

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4096
epoch = 200
lr = 3e-5

# Latent factor dim
print("layers = [128,128,128,128,128],lr = 3e-5")
layers = [128,128,128,128,128]
# 评价指标
n = 5


# Data Preparation

data_dir = '/root/autodl-fs/SDSC4016_Project/data'

train = pd.read_csv(os.path.join(data_dir, "senti_enhance_train.csv")).iloc[:, :3]

count = train.groupby(['uid', 'iid']).size().reset_index(name='count')
merged_df = pd.merge(train.copy(), count, on=['uid', 'iid'], how='inner')
merged_df['rating'] /= merged_df['count']
train1 = merged_df.iloc[:, :3]

test = pd.read_csv(os.path.join(data_dir, "test.csv")).iloc[:, :3]
test_like = like_generator(test)

# 用户u对应他访问过的所有items集合
train_user_items = train.groupby('uid')['iid'].apply(lambda x: np.array(x)).to_dict()

test_user_items = test.groupby('uid')['iid'].apply(lambda x: np.array(x)).to_dict()
test_like_user_items = test_like.groupby('uid')['iid'].apply(lambda x: np.array(x)).to_dict()

# 创建训练集张量
train_data = torch.tensor(train[['uid', 'iid']].values, dtype=torch.long).to(device)
train1_data= torch.tensor(train1[['uid', 'iid']].values, dtype=torch.long).to(device)
train_targets = torch.tensor(train['rating'].values, dtype=torch.float).to(device)
train1_targets = torch.tensor(train1['rating'].values, dtype=torch.float).to(device)

# 创建测试集张量
test_data = torch.tensor(test[['uid', 'iid']].values, dtype=torch.long).to(device)
test_targets = torch.tensor(test['rating'].values, dtype=torch.float).to(device)

# 使用 TensorDataset 封装数据
train_dataset = TensorDataset(train_data, train_targets)
test_dataset = TensorDataset(test_data, test_targets)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Training

train_statistic_log = ""
pre_train=True
model = DMF(NUM_USERS, NUM_ITEMS, layers, train1_data, train1_targets).to(device)
model.device = device
# MF model
if pre_train:
    model.load_state_dict(torch.load('DMF-best_model.pth'))




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
    
    msg = f'Train | [{x+1}/{epoch}], Loss: {train_loss/len(train_loader)}\n'
    print(msg)
    train_statistic_log += msg

    
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
    
    msg = f'Valid | [{x+1}/{epoch}], Loss: {val_loss/len(test_loader)}, Pre@{n}: {pre}, Rec@{n}: {rec}, NDCG@{n}: {ndcg}\n---------------------------------------------------\n'
    train_statistic_log += msg
    print(msg)

with open("log.txt", "w", encoding='utf-8') as f:
    f.write(train_statistic_log)
    f.close()