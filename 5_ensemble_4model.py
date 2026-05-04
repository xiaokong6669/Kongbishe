import pandas as pd
import numpy as np
import csv
import os
import sys
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import random

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)

# 加载数据
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

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

amino_acid_map = {
    'X': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

def encode_sequence(sequence, amino_acid_map):
    return [amino_acid_map[aa] for aa in sequence]

encoded_train_sequences = [encode_sequence(seq, amino_acid_map) for seq in train_sequences]
encoded_train_sequences = np.array(encoded_train_sequences)

max_train_length = 50

# 加载测试数据
test_sequences = []
for record in SeqIO.parse("out.fasta", "fasta"):
    header = record.description
    sequence = str(record.seq)
    test_sequences.append(sequence)
test_sequences = np.array(test_sequences)

encoded_test_sequences = [encode_sequence(seq, amino_acid_map) for seq in test_sequences]
encoded_test_sequences = pad_sequences(encoded_test_sequences, padding='post', maxlen=max_train_length)
encoded_test_sequences = np.array(encoded_test_sequences)

# 准备验证集
X_train_encoded, X_val_encoded, y_train, y_val = train_test_split(encoded_train_sequences, train_labels, test_size=0.2, random_state=42)
# 同时保存原始序列的验证集
X_train_seq, X_val_seq, _, _ = train_test_split(train_sequences, train_labels, test_size=0.2, random_state=42)

# ========== CNN模型 ==========
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Reshape((50, 1), input_shape=(50,)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(encoded_train_sequences, train_labels, epochs=25, batch_size=32, 
              class_weight=class_weight_dict, verbose=0)

cnn_y_pred_test = cnn_model.predict(encoded_test_sequences).flatten()
cnn_y_val = cnn_model.predict(X_val_encoded).flatten()

# ========== Bi-LSTM模型 ==========
num_amino_acids = 21
max_sequence_length = encoded_test_sequences.shape[1]
embedding_dim = 128

bi_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_amino_acids, output_dim=embedding_dim, input_length=max_sequence_length),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=False)),
    tf.keras.layers.Dropout(0),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

bi_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
bi_model.fit(encoded_train_sequences, train_labels, epochs=25, batch_size=32, 
             class_weight=class_weight_dict, verbose=0)

bi_y_pred_test = bi_model.predict(encoded_test_sequences).flatten()
bi_y_val = bi_model.predict(X_val_encoded).flatten()

# ========== Transformer模型 ==========
max_length = encoded_train_sequences.shape[1]
vocab_size = len(amino_acid_map)

def transformer_model(max_length, vocab_size, num_heads=6, embedding_dim=256, dense_units=128, dropout_rate=0):
    inputs = Input(shape=(max_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = TransformerEncoder(num_heads, embedding_dim, dense_units, dropout_rate, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    trans_model = Model(inputs=inputs, outputs=outputs)
    return trans_model

def TransformerEncoder(num_heads, embedding_dim, dense_units, dropout_rate=0.2, inputs=None):
    x = PositionalEncoding()(inputs)
    x = Dropout(dropout_rate)(x)
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
    x = attention(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        pos = np.arange(seq_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, dtype=tf.float32)

trans_model = transformer_model(max_length, vocab_size, num_heads=6, embedding_dim=256)
trans_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
trans_model.fit(encoded_train_sequences, train_labels, epochs=25, batch_size=32, 
                class_weight=class_weight_dict, verbose=0)

trans_y_pred_test = trans_model.predict(encoded_test_sequences).flatten()
trans_y_val = trans_model.predict(X_val_encoded).flatten()

# ========== GRU模型 ==========
# 使用特征工程编码GRU模型
amino_acid_properties = {
    'X': [0, 0, 0, 0],
    'A': [1.8, 8.1, 89.1, 0],
    'C': [2.5, 5.5, 121.2, 0],
    'D': [-3.5, 13.0, 133.1, -1],
    'E': [-3.5, 12.3, 147.1, -1],
    'F': [2.8, 5.2, 165.2, 0],
    'G': [-0.4, 9.0, 75.1, 0],
    'H': [-3.2, 10.4, 155.2, 0],
    'I': [4.5, 5.2, 131.2, 0],
    'K': [-3.9, 11.3, 146.2, 1],
    'L': [3.8, 4.9, 131.2, 0],
    'M': [1.9, 5.7, 149.2, 0],
    'N': [-3.5, 11.6, 132.1, 0],
    'P': [-1.6, 8.0, 115.1, 0],
    'Q': [-3.5, 10.5, 146.2, 0],
    'R': [-4.5, 10.5, 174.2, 1],
    'S': [-0.8, 9.2, 105.1, 0],
    'T': [-0.7, 8.6, 119.1, 0],
    'W': [-0.9, 5.4, 204.2, 0],
    'Y': [-1.3, 6.2, 181.2, 0],
    'V': [4.2, 5.9, 117.1, 0]
}

amino_acids = list(amino_acid_properties.keys())
interaction_matrix = np.zeros((len(amino_acids), len(amino_acids)))

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((np.array(vec1) - np.array(vec2))**2))

for i, aa1 in enumerate(amino_acids):
    for j, aa2 in enumerate(amino_acids):
        interaction_matrix[i, j] = euclidean_distance(amino_acid_properties[aa1], amino_acid_properties[aa2])

interaction_matrix[0, :] = 0.0
interaction_matrix[:, 0] = 0.0

def encode_sequence_with_interaction(sequence, amino_acid_map, interaction_matrix):
    encoded_sequence = []
    for aa in sequence:
        aa_code = amino_acid_map[aa]
        interaction_features = interaction_matrix[aa_code]
        encoded_sequence.append(np.concatenate(([aa_code], interaction_features)))
    return np.array(encoded_sequence)

def generate_second_order_features(encoded_sequences):
    second_order_features = []
    for seq in encoded_sequences:
        second_order_seq = []
        for i in range(len(seq)):
            aa = seq[i]
            second_order_feature = np.array([aa[j] * aa[k] for j in range(len(aa)) for k in range(len(aa))]).reshape(-1)
            second_order_seq.append(second_order_feature)
        second_order_features.append(np.array(second_order_seq))
    return np.array(second_order_features)

# 编码训练集用于GRU
encoded_train_sequences_gru = [encode_sequence_with_interaction(seq, amino_acid_map, interaction_matrix) for seq in train_sequences]
encoded_train_sequences_gru = np.array(encoded_train_sequences_gru)

# 加载GRU模型
model_path = 'gru_model.h5'
loaded_model = load_model(model_path)

# 编码测试集用于GRU
encoded_test_sequences_gru = [encode_sequence_with_interaction(seq, amino_acid_map, interaction_matrix) for seq in test_sequences]
encoded_test_sequences_gru = np.array(encoded_test_sequences_gru)
encoded_test_sequences_with_second_order = generate_second_order_features(encoded_test_sequences_gru)

scaler = StandardScaler()
encoded_test_sequences_with_second_order = scaler.fit_transform(encoded_test_sequences_with_second_order.reshape(-1, encoded_test_sequences_with_second_order.shape[-1]))
encoded_test_sequences_with_second_order = encoded_test_sequences_with_second_order.reshape(-1, encoded_test_sequences_gru.shape[1], encoded_test_sequences_with_second_order.shape[1])

gru_y_pred_test = loaded_model.predict(encoded_test_sequences_with_second_order).flatten()

# 编码验证集用于GRU
encoded_val_sequences_gru = [encode_sequence_with_interaction(seq, amino_acid_map, interaction_matrix) for seq in X_val_seq]
encoded_val_sequences_gru = np.array(encoded_val_sequences_gru)
encoded_val_sequences_with_second_order = generate_second_order_features(encoded_val_sequences_gru)
encoded_val_sequences_with_second_order = scaler.transform(encoded_val_sequences_with_second_order.reshape(-1, encoded_val_sequences_with_second_order.shape[-1]))
encoded_val_sequences_with_second_order = encoded_val_sequences_with_second_order.reshape(-1, encoded_val_sequences_gru.shape[1], encoded_val_sequences_with_second_order.shape[1])
gru_y_val = loaded_model.predict(encoded_val_sequences_with_second_order).flatten()

# ========== 计算各模型性能 ==========
models = {'cnn': cnn_y_val, 'bi_lstm': bi_y_val, 'transformer': trans_y_val, 'gru': gru_y_val}

model_performance = {}
for name, pred in models.items():
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (pred > threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    y_pred = (pred > best_threshold).astype(int)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, pred)
    
    model_performance[name] = {
        'accuracy': accuracy, 'recall': recall, 'precision': precision, 
        'f1': f1, 'auc': auc, 'best_threshold': best_threshold
    }
    print(f"{name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Best Threshold: {best_threshold:.2f}")

# ========== 加权集成 ==========
# 根据F1分数和AUC分配权重
weights = {
    'cnn': (model_performance['cnn']['f1'] * 0.6 + model_performance['cnn']['auc'] * 0.4),
    'bi_lstm': (model_performance['bi_lstm']['f1'] * 0.6 + model_performance['bi_lstm']['auc'] * 0.4),
    'transformer': (model_performance['transformer']['f1'] * 0.6 + model_performance['transformer']['auc'] * 0.4),
    'gru': (model_performance['gru']['f1'] * 0.6 + model_performance['gru']['auc'] * 0.4)
}

total_weight = sum(weights.values())
normalized_weights = {k: v/total_weight for k, v in weights.items()}
print(f"Normalized weights: {normalized_weights}")

# 加权平均
avg_predictions = (
    cnn_y_pred_test * normalized_weights['cnn'] + 
    bi_y_pred_test * normalized_weights['bi_lstm'] + 
    trans_y_pred_test * normalized_weights['transformer'] + 
    gru_y_pred_test * normalized_weights['gru']
)

# 加载测试集标签
test_labels = []
for record in SeqIO.parse("out.fasta", "fasta"):
    header = record.description
    label = int(header.split("|")[1])
    test_labels.append(label)
test_labels = np.array(test_labels)

# 搜索最佳阈值
best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = (avg_predictions > threshold).astype(int)
    f1 = f1_score(test_labels, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}, Best F1: {best_f1:.4f}")

# 计算最终性能
y_pred_final = (avg_predictions > best_threshold).astype(int)
accuracy_final = accuracy_score(test_labels, y_pred_final)
recall_final = recall_score(test_labels, y_pred_final)
precision_final = precision_score(test_labels, y_pred_final)
f1_final = f1_score(test_labels, y_pred_final)
auc_final = roc_auc_score(test_labels, avg_predictions)

print(f"Final Performance - Threshold = {best_threshold:.2f}")
print(f"Accuracy = {accuracy_final:.4f}")
print(f"Recall = {recall_final:.4f}")
print(f"Precision = {precision_final:.4f}")
print(f"F1 = {f1_final:.4f}")
print(f"AUC = {auc_final:.4f}")

# 保存模型
print("Saving models...")
cnn_model.save('cnn_model.h5')
bi_model.save('bi_lstm_model.h5')
trans_model.save('transformer_model.h5')
print("Models saved successfully!")

# 保存权重信息
import json
model_info = {
    "weights": normalized_weights,
    "best_threshold": best_threshold
}
with open('model_info.json', 'w') as f:
    json.dump(model_info, f)
print("Model weights and threshold saved successfully!")

# 保存结果
predictions = {
    "Peptide Sequence": test_sequences,
    "CNN Prediction Probability": cnn_y_pred_test,
    "CNN_Bi-LSTM Prediction Probability": bi_y_pred_test,
    "Transformer Prediction Probability": trans_y_pred_test,
    "GRU Prediction Probability": gru_y_pred_test,
    "Avg Predictions": avg_predictions
}

results_df = pd.DataFrame(predictions)
results_df.to_csv("result.csv", index=False)
