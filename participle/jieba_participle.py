# -*- coding: utf-8 -*-

import jieba
from pathlib import Path

# 读取文本文件
with open('../paChong/assets/output.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 使用 jieba 分词
words = jieba.cut(text)

# 将分词结果转换为列表
word_list = list(words)

# 保存分词结果到文件
output_path = Path('assets/jieba.txt')
output_path.write_text("\n".join(word_list), "utf-8")
