# -*- coding: utf-8 -*-
from transformers import BertTokenizer
from pathlib import Path

# 读取文本文件
def read_text(file_path: str) -> str:
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


# 保存分词结果
def save_words(words: list[str], output_file: str) -> None:
    path = Path(output_file)
    path.write_text("\n".join(words), "utf-8")


# 主函数
def main():
    # 读取文本文件
    input_file_path = '../paChong/assets/output.txt'  # 请替换为实际的输入文件路径
    text = read_text(input_file_path)

    # 使用 BERT 分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    print(tokenizer)
    # 分词
    tokens = tokenizer.tokenize(text)

    # 输出分词结果到文件
    output_file_path = './bert_tokens.txt'  # 请替换为实际的输出文件路径
    save_words(tokens, output_file_path)

    # 打印分词结果
    print("分词结果：")
    print(tokens)


if __name__ == "__main__":
    main()
