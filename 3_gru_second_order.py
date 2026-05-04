import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.data import BatchLoader
from tqdm import tqdm
from Bio import SeqIO
import networkx as nx
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, GlobalAveragePooling1D, LayerNormalization
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization,Dropout,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import scipy.sparse as sp0
from typing import Optional
from spektral.utils import normalized_adjacency
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler


amino_acid_properties = {
    'X': [0, 0, 0, 0],
    'A': [1.8, 8.1, 89.1, 0],   # Alanine
    'C': [2.5, 5.5, 121.2, 0],   # Cysteine
    'D': [-3.5, 13.0, 133.1, -1], # Aspartic acid
    'E': [-3.5, 12.3, 147.1, -1], # Glutamic acid
    'F': [2.8, 5.2, 165.2, 0],   # Phenylalanine
    'G': [-0.4, 9.0, 75.1, 0],    # Glycine
    'H': [-3.2, 10.4, 155.2, 0],  # Histidine
    'I': [4.5, 5.2, 131.2, 0],   # Isoleucine
    'K': [-3.9, 11.3, 146.2, 1],  # Lysine
    'L': [3.8, 4.9, 131.2, 0],   # Leucine
    'M': [1.9, 5.7, 149.2, 0],   # Methionine
    'N': [-3.5, 11.6, 132.1, 0],  # Asparagine
    'P': [-1.6, 8.0, 115.1, 0],   # Proline
    'Q': [-3.5, 10.5, 146.2, 0],  # Glutamine
    'R': [-4.5, 10.5, 174.2, 1],  # Arginine
    'S': [-0.8, 9.2, 105.1, 0],  # Serine
    'T': [-0.7, 8.6, 119.1, 0],  # Threonine
    'W': [-0.9, 5.4, 204.2, 0],   # Tryptophan
    'Y': [-1.3, 6.2, 181.2, 0],    # Tyrosine
    'V': [4.2, 5.9, 117.1, 0]    # Valine
}

amino_acids = list(amino_acid_properties.keys())

# 计算欧氏距离的函数
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2))**2))

# 初始化互作矩阵
interaction_matrix = np.zeros((len(amino_acids), len(amino_acids)))

# 计算每对氨基酸之间的互作
for i, aa1 in enumerate(amino_acids):
    for j, aa2 in enumerate(amino_acids):
        interaction_matrix[i, j] = euclidean_distance(amino_acid_properties[aa1], amino_acid_properties[aa2])

interaction_matrix[0, :] = 0.0   # 第 0 行
interaction_matrix[:, 0] = 0.0   # 第 0 列
#print(interaction_matrix.shape)

amino_acid_map = {
    'X': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

train_sequences = []
train_labels = []
for record in SeqIO.parse("Train_datasets_final.fasta", "fasta"):
    header = record.description
    sequence = str(record.seq)
    label = int(header.split("|")[1])
    train_sequences.append(sequence)
    train_labels.append(label)
train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)

# 编码函数：除了基本的氨基酸编码，还将互作矩阵的特征拼接到每个氨基酸的特征上
def encode_sequence_with_interaction(sequence, amino_acid_map, interaction_matrix):
    encoded_sequence = []
    for aa in sequence:
        # 获取氨基酸的基本编码
        aa_code = amino_acid_map[aa]
        # 获取氨基酸的互作特征（从矩阵中获取对应行）
        interaction_features = interaction_matrix[aa_code]  # 获取该氨基酸的互作特征
        # 将氨基酸的编码与互作特征拼接成一个新特征向量
        encoded_sequence.append(np.concatenate(([aa_code], interaction_features)))  # 拼接
    return np.array(encoded_sequence)

# 编码训练集
encoded_train_sequences = [encode_sequence_with_interaction(seq, amino_acid_map, interaction_matrix) for seq in train_sequences]
encoded_train_sequences = np.array(encoded_train_sequences)

max_train_length = 50

test_sequences = []
test_labels = []
for record in SeqIO.parse("test_21-30.fasta", "fasta"):
    header = record.description
    sequence = str(record.seq)
    label = int(header.split("|")[1])
    test_sequences.append(sequence)
    test_labels.append(label)
test_sequences = np.array(test_sequences)
test_labels = np.array(test_labels)

encoded_test_sequences = [encode_sequence_with_interaction(seq, amino_acid_map, interaction_matrix) for seq in test_sequences]
encoded_test_sequences = np.array(encoded_test_sequences)

# 修改 generate_second_order_features 函数，确保它返回的是三维数组
def generate_second_order_features(encoded_sequences):
    second_order_features = []
    for seq in encoded_sequences:
        second_order_seq = []
        for i in range(len(seq)):
            aa = seq[i]  # 获取该氨基酸的特征向量
            second_order_feature = np.array([aa[j] * aa[k] for j in range(len(aa)) for k in range(len(aa))]).reshape(-1)
            second_order_seq.append(second_order_feature)
        second_order_features.append(np.array(second_order_seq))
    return np.array(second_order_features)

# 生成训练集和测试集的二次特征
encoded_train_sequences_with_second_order = generate_second_order_features(encoded_train_sequences)
encoded_test_sequences_with_second_order = generate_second_order_features(encoded_test_sequences)

# 对数据进行标准化
scaler = StandardScaler()
encoded_train_sequences_with_second_order = scaler.fit_transform(encoded_train_sequences_with_second_order.reshape(-1, encoded_train_sequences_with_second_order.shape[-1]))
encoded_train_sequences_with_second_order = encoded_train_sequences_with_second_order.reshape(-1, encoded_train_sequences.shape[1], encoded_train_sequences_with_second_order.shape[1])

encoded_test_sequences_with_second_order = scaler.transform(encoded_test_sequences_with_second_order.reshape(-1, encoded_test_sequences_with_second_order.shape[-1]))
encoded_test_sequences_with_second_order = encoded_test_sequences_with_second_order.reshape(-1, encoded_test_sequences.shape[1], encoded_test_sequences_with_second_order.shape[1])


def gru_model(max_length, feat_dim, embedding_dim=256, dense_units=128, dropout_rate=0.5):
    inputs = Input(shape=(max_length, feat_dim))  # 输入层：max_length 为序列长度，feat_dim 为每个时间步的特征维度

    # GRU 网络，增加层数，加入 Dropout 和 L2 正则化
    x = GRU(512, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)  # 第一层 GRU
    x = Dropout(dropout_rate)(x)
    
    x = GRU(256, return_sequences=True, kernel_regularizer=l2(0.01))(x)  # 第二层 GRU
    x = Dropout(dropout_rate)(x)
    
    x = GRU(128, return_sequences=True, kernel_regularizer=l2(0.01))(x)  # 第三层 GRU
    x = Dropout(dropout_rate)(x)

    # 池化层，用于汇总信息
    x = GlobalAveragePooling1D()(x)  # 现在 GRU 返回整个序列的输出，所以 GlobalAveragePooling1D 可以处理

    # 全连接层：加入更多隐藏单元
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # 输出层
    outputs = Dense(1, activation='sigmoid')(x)

    # 构建并返回模型
    model = Model(inputs=inputs, outputs=outputs)
    return model

# 构建 GRU 模型
max_train_length = encoded_train_sequences.shape[1]  # 序列长度
feat_dim = encoded_train_sequences.shape[2]  # 特征维度（22）

# 因为二次特征增加，输入维度是 484
gru_model_instance = gru_model(max_train_length, feat_dim * 22, embedding_dim=256, dense_units=128, dropout_rate=0.5)  # 输入是 22 × 2

# 编译模型
gru_model_instance.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

# 训练模型
gru_model_instance.fit(encoded_train_sequences_with_second_order, train_labels, epochs=25, batch_size=32, verbose=1)

gru_model_instance.save('gru_model.h5')

gru_model_instance = load_model('gru_model.h5')

# 预测
gru_y_pred_test = gru_model_instance.predict(encoded_test_sequences_with_second_order)
gru_y_pred_flattened = gru_y_pred_test.flatten()


avg_predictions = gru_y_pred_flattened 

# Prepare results for CSV output
predictions = {
    "Peptide Sequence": test_sequences,
    "MLP Prediction Probability": gru_y_pred_flattened,
    "Avg Predictions": avg_predictions
}

# Save to CSV
results_df = pd.DataFrame(predictions)
results_df.to_csv("mlp.csv", index=False)












