# -*- coding: utf-8 -*-
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # 导入 tqdm 库

# 设置设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载训练数据
def load_data(file_path):
    sentences, labels = [], []
    sentence, label = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                char, tag = line.strip().split()
                sentence.append(char)
                label.append(tag)
            else:
                sentences.append(sentence)
                labels.append(label)
                sentence, label = [], []

    if sentence:  # 确保最后一句话被添加
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels


train_sentences, train_labels = load_data('./assets/LSTM_train.txt')

# 构建字符和标签的映射
chars = set([char for sentence in train_sentences for char in sentence])
tags = set([tag for label in train_labels for tag in label])

char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 字符映射为索引
tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}  # 标签映射为索引
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}  # 索引映射回标签

# 设置填充标签和字符
pad_tag = len(tag_to_idx)
char_to_idx["<PAD>"] = 0  # Padding token for characters
tag_to_idx["<PAD>"] = pad_tag  # Padding token for tags

# 将字符和标签转换为索引
X_train = [[char_to_idx[char] for char in sentence] for sentence in train_sentences]
y_train = [[tag_to_idx[tag] for tag in label] for label in train_labels]

# 填充序列
max_sequence_length = max(len(sentence) for sentence in X_train)


def pad_sequence(sequences, max_len, padding_value=0):
    return [seq + [padding_value] * (max_len - len(seq)) for seq in sequences]


X_train = pad_sequence(X_train, max_sequence_length)
y_train = pad_sequence(y_train, max_sequence_length, padding_value=pad_tag)

# 转换为Tensor并移动到GPU
X_train = torch.tensor(X_train, dtype=torch.long).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)


# 定义数据集
class CWSDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


train_dataset = CWSDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 定义双向LSTM模型
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=64):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.fc(lstm_out)
        return lstm_out


# 实例化模型并移动到GPU
model = BiLSTMTagger(vocab_size=len(char_to_idx), tagset_size=len(tag_to_idx)).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=pad_tag)  # 忽略填充部分的标签
optimizer = torch.optim.Adadelta(model.parameters())


# 训练模型
def train_model(model, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # 在每个 epoch 中使用 tqdm 包装数据加载器
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for sentences, labels in progress_bar:
            # 将数据移动到GPU
            sentences = sentences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sentences)

            # 将outputs和labels转换为合适的形状
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 更新进度条后面的损失值
            progress_bar.set_postfix(loss=total_loss)

        print(f"Epoch {epoch + 1} completed, Loss: {total_loss:.4f}")


train_model(model, train_loader, epochs=60)


# 加载要分词的txt文本
def load_test_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return list(text)


test_text = load_test_text('../paChong/assets/output.txt')

# 将测试文本转换为索引
X_test = [[char_to_idx.get(char, 0) for char in test_text]]  # OOV词使用0表示
X_test = pad_sequence(X_test, max_sequence_length)

X_test = torch.tensor(X_test, dtype=torch.long).to(device)  # 移动到GPU

# 模型预测
# 开始计时
start_time = time.time()
with torch.no_grad():
    predictions = model(X_test)

# 取出最大概率的标签索引
predicted_tags = torch.argmax(predictions, dim=-1).squeeze().tolist()


# 将预测结果转换回标签
def decode_predictions(sentence, predicted_tags):
    result = []
    word = ""
    for char, tag_idx in zip(sentence, predicted_tags):
        tag = idx_to_tag[tag_idx]
        if tag == 'B-CWS':
            if word:
                result.append(word)
            word = char
        elif tag == 'I-CWS':
            word += char
        elif tag == 'E-CWS':
            word += char
            result.append(word)
            word = ""
        elif tag == 'S-CWS':
            result.append(char)
            word = ""
    if word:
        result.append(word)
    return result


segmented_text = decode_predictions(test_text, predicted_tags)
# 结束计时
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Text segmentation completed in {elapsed_time:.4f} seconds.")

# 将分词结果保存到文件
with open('./assets/segmented_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(segmented_text))
