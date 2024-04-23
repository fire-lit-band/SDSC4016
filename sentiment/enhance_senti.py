# read the origin data
import pandas as pd
train= pd.read_csv('/root/autodl-fs/SDSC4016_Project/data/train_with_index.csv',index_col=0)

train=train[['uid','iid','rating']]
train.dropna(inplace=True)
print(train.shape)
train.index=train.index.astype(int)
print(train.shape)

train=train.sort_index()
train=train.astype({'uid': 'int32', 'iid': 'int32', 'rating': 'float32'})
# origin=pd.read_csv('/root/autodl-fs/data.csv')
# print(origin[ origin['user_id']==47823]) #origin['business_id']==0 &
prob= pd.read_csv('/root/autodl-fs/prob.csv')
print(train.head())





prob=prob.rename(columns={'user_id': 'uid', 'business_id': 'iid'})
prob=prob[['prob','uid','iid']] #,
print(prob.head())
prob=prob.astype({ 'uid': 'int32', 'iid': 'int32','prob': 'float32'})#
print(prob.shape)
# 使用 'uid' 和 'iid' 这两列来合并 df 和 df1
merged_df = train.merge(prob,left_index=True, right_index=True)
merged_df = merged_df.drop(columns=['uid_x', 'iid_x'])
merged_df = merged_df.rename(columns={'uid_y': 'uid', 'iid_y': 'iid'})
print(merged_df.head())
print(merged_df.shape)




# 将合并后的 DataFrame 保存为新的 CSV 文件


alpha=2
print(f"alpha={alpha}")

# 计算加权后的评分
merged_df['rating'] = alpha* merged_df['rating'] +(1-alpha) * (4*merged_df['prob']+1)
# merged_df['rating'] = merged_df['rating'] +alpha * merged_df['prob']
new_column_order=['uid', 'iid', 'rating','prob']
merged_df = merged_df[new_column_order]
print(merged_df.head())
merged_df.to_csv('/root/autodl-fs/SDSC4016_Project/data/senti_enhance_train.csv', index=False)

