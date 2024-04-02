#!/usr/bin/env python
# coding: utf-8


# In[8]:
from collections import defaultdict
import json
import csv



# In[13]:


with open('../data/raw/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
    data_list = []
    for line in f:
        try:
            data = json.loads(line)
            # 只保留部分键
            selected_data = {
                "business_id": data["business_id"],
                "user_id":data["user_id"],



                "stars": data["stars"],
                "text":data["text"],
                "date":data["date"]


            }
            data_list.append(selected_data)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)


len(data_list)


# In[12]:

id_mapping = {}
encoded_id = 0
id_mapping_dict = {}  # 创建一个空字典用于存储 original_id 和 encoded_id 的对应关系

# 遍历数据列表，将business_id转换为整数编码
for item in data_list:
    original_id = item['business_id']
    # 检查是否已经存在相同的business_id
    if original_id not in id_mapping:
        id_mapping[original_id] = encoded_id
        item['business_id'] = encoded_id
        id_mapping_dict[original_id] = encoded_id  # 更新字典，存储对应关系
        encoded_id += 1
    else:
        # 如果已经存在相同的business_id，则保留原始的整数编码
        item['business_id'] = id_mapping[original_id]

    # 添加键值对，表示原始的business_id
    item['original_business_id'] = original_id



print(encoded_id)

# 打印 original_id 和 encoded_id 的对应关系字典



# In[ ]:




id_mapping_user = {}
encoded_id_user = 0
id_mapping_dict_user = {}
# 遍历数据列表，将business_id转换为整数编码
for item in data_list:
    original_id = item['user_id']
    # 检查是否已经存在相同的business_id
    if original_id not in id_mapping_user:
        id_mapping_user[original_id] = encoded_id_user
        item['user_id'] = encoded_id_user
        id_mapping_dict_user[original_id] = encoded_id_user
        encoded_id_user += 1
    else:
        # 如果已经存在相同的business_id，则保留原始的整数编码
        item['user_id'] = id_mapping_user[original_id]
    item['original_user_id'] = original_id

# 打印转换后的数据列表
print(encoded_id_user)


# %%

# %%
for i in range(100):
    print(data_list[i])
# %%
def item_filter(data_list,thresh):
    item_id_count = defaultdict(int)
    for entry in data_list:
        if 'user_id' in entry and isinstance(entry['business_id'], int):  # 检查user_id是否存在并且是整数类型
            item_id_count[entry['business_id']] += 1

# 筛选出超过20次的user_id的条目
    filtered_data = [entry for entry in data_list if item_id_count.get(entry.get('business_id', -1), 0) > thresh]
    count_over = sum(1 for count in item_id_count.values() if count > thresh)
    print("数量:", count_over)
    return filtered_data


def user_filter(filtered_data,thresh):
    user_id_count = defaultdict(int)
    for entry in filtered_data:
        if 'user_id' in entry and isinstance(entry['user_id'], int):  # 检查user_id是否存在并且是整数类型
            user_id_count[entry['user_id']] += 1

    filtered_data_2= [entry for entry in data_list if user_id_count.get(entry.get('user_id', -1), 0) > thresh]
    count_over = sum(1 for count in user_id_count.values() if count > thresh)
    print("数量:", count_over)
    return filtered_data_2




# %%
filtered=item_filter(data_list,20)
# %%
filtered_2=user_filter(filtered,20)
# %%
print(len(filtered_2))
# %%

id_mapping_2 = {}
encoded_id = 0
id_mapping_dict_2 = {} 
for item in filtered_2:
    original_id = item['original_business_id']
    # 检查是否已经存在相同的business_id
    if original_id not in id_mapping_2:
        id_mapping_2[original_id] = encoded_id
        item['business_id'] = encoded_id
        id_mapping_dict_2[original_id] = encoded_id  # 更新字典，存储对应关系
        encoded_id += 1
    else:
        # 如果已经存在相同的business_id，则保留原始的整数编码
        item['business_id'] = id_mapping_2[original_id]

    # 添加键值对，表示原始的business_id
    item['business_id_before'] = original_id


id_mapping_user_2 = {}
encoded_id_user = 0
id_mapping_dict_user_2 = {} 

for item in filtered_2:
    original_id = item['original_user_id']
    # 检查是否已经存在相同的business_id
    if original_id not in id_mapping_user_2:
        id_mapping_user_2[original_id] = encoded_id_user
        item['user_id'] = encoded_id_user
        id_mapping_dict_user_2[original_id] = encoded_id_user
        encoded_id_user += 1
    else:
        # 如果已经存在相同的business_id，则保留原始的整数编码
        item['user_id'] = id_mapping_user_2[original_id]
    item['user_id_before'] = original_id

# 打印转换后的数据列表
    

for i in range(100):
    print(filtered_2[i])
# %%



with open('../data/raw/yelp_academic_dataset_user.json', 'r', encoding='utf-8') as f:
    user_list = []
    for line in f:
        try:
            data = json.loads(line)
            # 只保留部分键
            selected_data = {

                "user_id":data["user_id"],
                "friends":data["friends"]




            }
            user_list.append(selected_data)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)




# %%
for i in range(100):
    print(user_list[i])
# %%
len(id_mapping_dict_user_2) 



# 打印字典的前100项

# %%
for user in user_list:
    # 如果'user_id'为空值，则跳过此用户
    if not user['user_id']:
        continue

    # 处理'friends'字段
    friends = user['friends'].split(', ')
    user['friends'] = [id_mapping_dict_user_2.get(friend) for friend in friends if id_mapping_dict_user_2.get(friend)]
    user['friends'] = [friend for friend in user['friends'] if friend] 
    # 处理'user_id'字段
    user_id = user['user_id']
    user['user_id'] = id_mapping_dict_user_2.get(user_id)





# %%
user_list = [user for user in user_list if user['user_id']]
len(user_list)
# %%
for i in range(100):
    print(user_list[i])
# %%
csv_file_path = 'user_list.csv'

# CSV文件的表头，即字段名
fieldnames = ['user_id', 'friends']

# 写入CSV文件

csv_file_path = 'data.csv'
fieldnames = ['business_id','user_id','stars','text','date','original_business_id','original_user_id','business_id_before','user_id_before']
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # 写入表头
    writer.writeheader()
    
    # 写入数据
    for user in filtered_2:
        writer.writerow(user)
# %%
