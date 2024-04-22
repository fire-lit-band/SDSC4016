import matplotlib.pyplot as plt

# 初始化存储数据的列表
train_losses = []
valid_losses = []
precisions_at_5 = []
recalls_at_5 = []
ndcgs_at_5 = []

# 读取文件并解析数据
with open('5x128lr1e-5.txt', 'r') as file:
    for line in file:
        if 'Train' in line:
            # 提取训练损失
            parts = line.split(',')
            loss = float(parts[1].split(':')[-1].strip())
            train_losses.append(loss)
        elif 'Valid' in line:
            # 提取验证损失和其他指标
            parts = line.split(',')
            valid_loss = float(parts[1].split(':')[-1].strip())
            pre_at_5 = float(parts[2].split(':')[-1].strip())
            rec_at_5 = float(parts[3].split(':')[-1].strip())
            ndcg_at_5 = float(parts[4].split(':')[-1].strip())

            valid_losses.append(valid_loss)
            precisions_at_5.append(pre_at_5)
            recalls_at_5.append(rec_at_5)
            ndcgs_at_5.append(ndcg_at_5)

# 创建图表
plt.figure(figsize=(14, 10))

# 绘制验证损失
plt.subplot(2, 2, 1)
plt.plot(valid_losses[5:], label='Validation Loss', color='blue', marker='o')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 绘制Precision@5
plt.subplot(2, 2, 2)
plt.plot(precisions_at_5[5:], label='Precision@5', color='green', marker='o')
plt.title('Precision@5')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()

# 绘制Recall@5
plt.subplot(2, 2, 3)
plt.plot(recalls_at_5[5:], label='Recall@5', color='red', marker='o')
plt.title('Recall@5')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.grid(True)
plt.legend()

# 绘制NDCG@5
plt.subplot(2, 2, 4)
plt.plot(ndcgs_at_5[5:], label='NDCG@5', color='purple', marker='o')
plt.title('NDCG@5')
plt.xlabel('Epochs')
plt.ylabel('NDCG')
plt.grid(True)
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
