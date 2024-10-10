# -*- coding: utf-8 -*-
import re


def remove_all_whitespace(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式删除所有空格、换行符和制表符
    content = re.sub(r'\s+', '', content)  # 匹配所有空白字符（空格、换行符、制表符等）

    # 将处理后的内容写入新的文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(content)


# 调用函数，指定输入和输出文件路径
remove_all_whitespace('assets/巫师秘旅-11-33.txt', 'paChong/output.txt')
