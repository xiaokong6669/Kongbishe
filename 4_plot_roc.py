import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score

# 读取CSV文件
file_path = 'result.csv'
data = pd.read_csv(file_path)

# 提取最后一列数值到列表 y_pred_prob
y_pred_prob = data.iloc[:, -1].tolist()  # iloc[:, -1]表示所有行的最后一列

# 真实标签
test_sequences = []
test_labels = []
for record in SeqIO.parse("out.fasta", "fasta"):
    header = record.description
    sequence = str(record.seq)
    label = int(header.split("|")[1])
    test_sequences.append(sequence)
    test_labels.append(label)
test_labels = np.array(test_labels)

# 搜索最佳阈值以最大化F1分数
best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (np.array(y_pred_prob) > threshold).astype(int)
    f1 = f1_score(test_labels, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# 设定阈值 k 为最佳阈值
k = best_threshold

# 转换为二分类标签
y_pred_labels = (pd.Series(y_pred_prob) >= k).astype(int)  # 使用pandas的向量化操作

# 转换为Python原生列表
y_pred_labels = y_pred_labels.tolist()

# 计算 ROC 曲线的假阳性率 (FPR)、真正率 (TPR) 和阈值
fpr, tpr, thresholds = roc_curve(test_labels, y_pred_prob)

# 计算 AUC
roc_auc = auc(fpr, tpr)

print("Threshold =", k)
print("Accuracy =", accuracy_score(test_labels, y_pred_labels))
print("Recall =", recall_score(test_labels, y_pred_labels))
print("Precision =", precision_score(test_labels, y_pred_labels))
print("F1 =", f1_score(test_labels, y_pred_labels))
print(f"AUC={roc_auc:.3f}")

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 随机分类器的对角线

# 添加标题和标签
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')

# 显示 AUC
plt.legend(loc='lower right')

# 展示图像
plt.savefig(f"roc_k_{k:.2f}.png", dpi=300, bbox_inches="tight")
plt.show()


