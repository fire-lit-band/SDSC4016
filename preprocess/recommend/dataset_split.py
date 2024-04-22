import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split(df, ratio):
    uid = list(set(df.iloc[:,0].values))
    train = pd.DataFrame(columns = df.columns, dtype=int)
    test = pd.DataFrame(columns = df.columns, dtype=int)
    for i in uid:
        train_1, test_1 = train_test_split(df[df.iloc[:, 0] == i], train_size = ratio, shuffle = True, random_state = 5)
        train = pd.concat([train, train_1])
        test = pd.concat([test, test_1])
    return train, test  

if __name__ == "__main__":
    train_ratio = 0.8
    
    data = pd.read_csv("./data.csv").iloc[:, :5]
    data = data.reset_index(drop=False)
    data['index'] = data['user_id']
    data = data.drop(columns=['user_id'])
    data = data.rename(columns = {data.columns[0]: 'uid', data.columns[1]: 'iid', data.columns[2]: 'rating'})
    print(f"Num of users: {len(list(set(data.iloc[:,0].values)))}\nNum of items: {len(list(set(data.iloc[:,1].values)))}")
    
    train, test = split(data, train_ratio)
    
    print(f"Train size: {train.shape[0]}\nTest size: {test.shape[0]}")
    

    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)