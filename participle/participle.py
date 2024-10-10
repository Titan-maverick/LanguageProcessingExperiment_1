# -*- coding: utf-8 -*-
import re
from pathlib import Path
import time

# 读取文本文件并保留标点符号
def read_text(file_path: str) -> str:
    path = Path(file_path)
    # 读取文本并去除换行
    text = path.read_text("utf-8")
    cleaned_text = re.sub(r'\s+', '', text)  # 去除所有空格和换行
    return cleaned_text

# 读取词典文件
def read_dictionary(file_path: str) -> set[str]:
    path = Path(file_path)
    return {line.split()[1] for line in path.read_text("gbk").splitlines()[1:]}

# 保存分词结果
def save_words(words: list[str], output_file: str) -> None:
    path = Path(output_file)
    path.write_text("\n".join(words), "utf-8")

# 正向最大匹配算法
def forward_maximum_matching(text: str, dictionary: set[str], max_length: int) -> list[str]:
    words = []
    i = 0
    while i < len(text):
        # 处理数字和小数
        if text[i].isdigit():
            num_start = i
            while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                i += 1
            words.append(text[num_start:i])  # 添加完整的数字或小数
            continue

        max_word = ""
        for j in range(i + 1, min(i + max_length + 1, len(text) + 1)):
            word = text[i:j]
            if word in dictionary and len(word) > len(max_word):
                max_word = word

        if max_word:
            words.append(max_word)
            i += len(max_word)
        else:
            # 处理单个字符或标点符号
            words.append(text[i])  # 直接将当前字符（包括标点符号）添加到结果中
            i += 1

    return words

# 逆向最大匹配算法
def backward_maximum_matching(text: str, dictionary: set[str], max_length: int) -> list[str]:
    words = []
    i = len(text)
    while i > 0:
        # 处理数字和小数
        if text[i - 1].isdigit():
            num_end = i
            while i > 0 and (text[i - 1].isdigit() or text[i - 2] == '.'):
                i -= 1
            words.insert(0, text[i:num_end])  # 添加完整的数字或小数
            continue

        max_word = ""
        for j in range(i - max_length, i):
            if j < 0:
                break
            word = text[j:i]
            if word in dictionary and len(word) > len(max_word):
                max_word = word
        if max_word:
            words.insert(0, max_word)
            i -= len(max_word)
        else:
            words.insert(0, text[i - 1])  # 处理单个字符或标点符号
            i -= 1
    return words

def get_longest_word_length(file_path: str) -> int:
    longest_length = 0
    file = read_dictionary(file_path)
    for line in file:
        word = line.strip()
        longest_length = max(longest_length, len(word))

    return longest_length

# 主函数示例
def main():
    file_path = '../wordlist.Dic'  # 请替换为实际文件路径
    longest_word_length = get_longest_word_length(file_path)

    print(f"最长词的长度是: {longest_word_length}")  # 打印最长词的长度

    text = read_text("../paChong/assets/output.txt")
    dictionary = read_dictionary("../wordlist.Dic")

    # 逆向最大匹配
    start_time = time.time()
    backward_words = backward_maximum_matching(text, dictionary, longest_word_length)
    backward_words = [word for word in backward_words if word.isalnum() or word.isdigit() or word in ['.', '、', '。', '，', '？', '！', '“', '”', '‘', '’', '；', '：', '—', '《', '》']]
    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time
    save_words(backward_words, "assets/backward_result.txt")
    print(f"逆向最大匹配耗时: {elapsed_time:.6f}秒")

    start_time = time.time()
    # 正向最大匹配
    forward_words = forward_maximum_matching(text, dictionary, longest_word_length)
    forward_words = [word for word in forward_words if word.isalnum() or word.isdigit() or word in ['.', '、', '。', '，', '？', '！', '“', '”', '‘', '’', '；', '：', '—', '《', '》']]
    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time
    save_words(forward_words, "assets/forward_result.txt")
    print(f"正向最大匹配耗时: {elapsed_time:.6f}秒")

if __name__ == "__main__":
    main()
