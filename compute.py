# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import List, Tuple


# 读取文本文件
def read_text(file_path: str) -> str:
    path = Path(file_path)
    return path.read_text("utf-8").strip()


# 读取分词结果
def read_words(file_path: str) -> List[str]:
    return read_text(file_path).splitlines()


# 计算 P, R, F 值
def calculate_metrics(standard: List[str], predicted: List[str]) -> Tuple[float, float, float]:
    # 计算正确预测的单词数
    tp = sum(1 for word in predicted if word in standard)
    # 计算预测错误的单词数
    fp = len(predicted) - tp
    # 计算标准答案中未出现的单词数
    fn = len(standard) - tp

    # 计算精确率
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    # 计算召回率
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    # 计算F1值
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score


# 主函数
def main():
    # 读取标准分词结果
    standard_words = read_words("participle/assets/jieba.txt")  # 替换为实际标准分词结果文件路径

    # 读取分词结果
    forward_words = read_words("participle/assets/forward_result.txt")
    backward_words = read_words("participle/assets/backward_result.txt")
    crf_words = read_words("CRF/assets/segmented_output.txt")
    lstm_words = read_words("LSTM_participle/assets/segmented_output.txt")

    # 计算指标
    forward_metrics = calculate_metrics(standard_words, forward_words)
    backward_metrics = calculate_metrics(standard_words, backward_words)
    crf_metrics = calculate_metrics(standard_words, crf_words)
    lstm_metrics = calculate_metrics(standard_words, lstm_words)

    # 打印结果
    print(f"正向最大匹配:\n  Precision: {forward_metrics[0]:.4f}\n  Recall: {forward_metrics[1]:.4f}\n  F1-score: {forward_metrics[2]:.4f}")
    print(f"逆向最大匹配:\n  Precision: {backward_metrics[0]:.4f}\n  Recall: {backward_metrics[1]:.4f}\n  F1-score: {backward_metrics[2]:.4f}")
    print(f"条件随机场:\n  Precision: {crf_metrics[0]:.4f}\n  Recall: {crf_metrics[1]:.4f}\n  F1-score: {crf_metrics[2]:.4f}")
    print(f"循环神经网络:\n  Precision: {lstm_metrics[0]:.4f}\n  Recall: {lstm_metrics[1]:.4f}\n  F1-score: {lstm_metrics[2]:.4f}")

    # 更换基准为LSTM, 进行测试
    standard_words = read_words("LSTM_participle/assets/segmented_output.txt")
    forward_metrics = calculate_metrics(standard_words, forward_words)
    backward_metrics = calculate_metrics(standard_words, backward_words)
    crf_metrics = calculate_metrics(standard_words, crf_words)
    lstm_metrics = calculate_metrics(standard_words, lstm_words)

    # 打印结果
    print("\n\n\n\n\n 更换基准")
    print(f"正向最大匹配:\n  Precision: {forward_metrics[0]:.4f}\n  Recall: {forward_metrics[1]:.4f}\n  F1-score: {forward_metrics[2]:.4f}")
    print(f"逆向最大匹配:\n  Precision: {backward_metrics[0]:.4f}\n  Recall: {backward_metrics[1]:.4f}\n  F1-score: {backward_metrics[2]:.4f}")
    print(f"条件随机场:\n  Precision: {crf_metrics[0]:.4f}\n  Recall: {crf_metrics[1]:.4f}\n  F1-score: {crf_metrics[2]:.4f}")
    print(f"循环神经网络:\n  Precision: {lstm_metrics[0]:.4f}\n  Recall: {lstm_metrics[1]:.4f}\n  F1-score: {lstm_metrics[2]:.4f}")

if __name__ == "__main__":
    main()
