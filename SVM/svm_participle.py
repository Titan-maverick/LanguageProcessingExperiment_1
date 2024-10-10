# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm  # 导入 tqdm
import time  # 导入 time

# 读取文件函数
def load_data(file_path):
    sentences = []
    labels = []
    sentence = []
    label = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                sentence = []
                label = []
            else:
                char, tag = line.strip().split()
                sentence.append(char)
                label.append(tag)

    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

train_sentences, train_labels = load_data('../chinese_word_segmentation_pku/train.txt')
test_sentences, test_labels = load_data('../chinese_word_segmentation_pku/test.txt')

# 提取特征函数
def extract_features(sentences):
    features = []
    for sentence in tqdm(sentences, desc="提取特征"):  # 显示进度条
        for i in range(len(sentence)):
            feature = {
                'char': sentence[i],
                'prev_char': sentence[i-1] if i > 0 else '',
                'next_char': sentence[i+1] if i < len(sentence) - 1 else ''
            }
            features.append(feature)
    return features

# 提取特征
train_features = extract_features(train_sentences)
test_features = extract_features(test_sentences)

# 使用稀疏矩阵进行特征向量化
vectorizer = DictVectorizer(sparse=True)  # 使用稀疏矩阵
X_train = vectorizer.fit_transform(train_features)
X_test = vectorizer.transform(test_features)

# 将标签转化为简单的数字编码
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(np.concatenate(train_labels))
y_test = label_encoder.transform(np.concatenate(test_labels))

# 训练SVM模型
svm_model = SVC(kernel='linear', decision_function_shape='ovr')

# 训练进度条
tqdm.write("正在训练模型...")
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 对长文本进行分词
def segment_text(text, model, vectorizer, label_encoder):
    sentences = list(text)
    features = extract_features([sentences])  # 提取特征
    X = vectorizer.transform(features)  # 向量化
    y_pred = model.predict(X)  # 预测
    predicted_labels = label_encoder.inverse_transform(y_pred)  # 转化为标签

    segmented_sentence = []
    for char, label in zip(sentences, predicted_labels):
        if label in ['B-CWS', 'S-CWS']:
            segmented_sentence.append(' ')
        segmented_sentence.append(char)

    return ''.join(segmented_sentence)

# 读取长文本
with open('../paChong/assets/output.txt', 'r', encoding='utf-8') as f:
    long_text = f.read()

# 记录分词开始时间
start_time = time.time()

# 分词
segmented_text = segment_text(long_text, svm_model, vectorizer, label_encoder)

# 记录分词结束时间
end_time = time.time()
segmentation_time = end_time - start_time

# 输出结果
print("分词结果:")
print(segmented_text)
print(f"分词时长: {segmentation_time:.4f}秒")

# 保存分词结果到文件
with open('../paChong/assets/segmented_output.txt', 'w', encoding='utf-8') as f:
    f.write(segmented_text)
