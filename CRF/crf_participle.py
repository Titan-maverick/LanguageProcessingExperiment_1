import re
from pathlib import Path
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
import time


def extract_features(chars, index):
    """提取字符特征"""
    char = chars[index]
    features = {
        'current_char': char,
        'is_first': index == 0,
        'is_last': index == len(chars) - 1,
        'prev_char': '' if index == 0 else chars[index - 1],
        'next_char': '' if index == len(chars) - 1 else chars[index + 1],
        'is_digit': char.isdigit(),
        'is_alpha': char.isalpha(),
        'is_chinese': bool(re.match(r'[\u4e00-\u9fa5]', char)),
        'is_punctuation': bool(re.match(r'[\u3000-\u303F\uFF00-\uFFEF]', char)),
    }
    return features


def extract_features_from_sentence(sentence):
    """从单个句子中提取特征"""
    chars = list(sentence)
    return [extract_features(chars, i) for i in range(len(chars))]


def read_text(file_path: str) -> list:
    """读取标注文本文件，返回句子和对应标签"""
    path = Path(file_path)
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # 非空行
                    char, label = line.strip().split()
                    current_sentence.append(char)
                    current_labels.append(label)
                else:  # 空行表示句子结束
                    if current_sentence:
                        sentences.append("".join(current_sentence))
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
            # 处理最后一个句子（如果没有空行结束）
            if current_sentence:
                sentences.append("".join(current_sentence))
                labels.append(current_labels)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except UnicodeDecodeError:
        print(f"文件编码错误: {file_path}")

    return sentences, labels


def train_crf_model(X_train, y_train):
    """训练CRF模型"""
    crf = CRF(algorithm='lbfgs', max_iterations=100)
    crf.fit(X_train, y_train)
    return crf


def evaluate_model(crf, X_test, y_test):
    """评估CRF模型"""
    y_pred = crf.predict(X_test)
    print("预测标签长度:", len(y_pred))
    print("真实标签长度:", len(y_test))
    print(flat_classification_report(y_test, y_pred))


def segment_text(crf, text):
    """使用训练好的模型对一段文本进行分词"""
    features = extract_features_from_sentence(text)
    pred = crf.predict([features])  # 预测结果
    return pred[0]  # 返回预测标签


def words_from_tags(chars, tags):
    """根据标签生成分词结果"""
    words = []
    current_word = []

    for char, tag in zip(chars, tags):
        if tag.startswith('B-'):  # 新词的开始
            if current_word:  # 如果有当前词，先保存
                words.append(''.join(current_word))
                current_word = []
            current_word.append(char)  # 开始新词
        elif tag.startswith('I-'):  # 继续当前词
            current_word.append(char)
        elif tag.startswith('E-'):  # 当前词结束
            current_word.append(char)
            words.append(''.join(current_word))
            current_word = []
        elif tag == 'S-CWS':  # 单个字符词
            words.append(char)

    # 处理剩余的词
    if current_word:
        words.append(''.join(current_word))

    return words


def save_segmented_text(words, output_file):
    """保存分词结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(f"{word}\n")  # 用换行符分割


def read_input_text(input_file: str) -> str:
    """读取待分词文本"""
    path = Path(input_file)
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read().strip()  # 读取并返回文本内容
    except FileNotFoundError:
        print(f"文件未找到: {input_file}")
        return ""


# 主函数
def main():
    # 读取训练数据
    train_sentences, train_labels = read_text("../chinese_word_segmentation_pku/train.txt")

    # 准备训练数据
    X_train = []
    y_train = []

    for i, sentence in enumerate(train_sentences):
        features = extract_features_from_sentence(sentence)
        X_train.append(features)
        y_train.append(train_labels[i])  # 确保 y_train 是嵌套列表

    # 训练模型
    crf_model = train_crf_model(X_train, y_train)

    # 读取测试数据
    test_sentences, test_labels = read_text("../chinese_word_segmentation_pku/test.txt")

    # 准备测试数据
    X_test = []
    y_test = []

    for i, sentence in enumerate(test_sentences):
        features = extract_features_from_sentence(sentence)
        X_test.append(features)
        y_test.append(test_labels[i])  # 确保 y_test 是嵌套列表

    # 评估模型
    evaluate_model(crf_model, X_test, y_test)

    # 开始计时
    start_time = time.time()
    # 处理待分词文本
    input_file = "../paChong/assets/output.txt"  # 待分词文本的文件名
    new_text = read_input_text(input_file)

    if new_text:  # 确保读取成功
        segmented_result = segment_text(crf_model, new_text)

        # 根据预测标签生成分词结果
        words = words_from_tags(list(new_text), segmented_result)

        # 保存分词结果
        output_file = "assets/segmented_output.txt"
        save_segmented_text(words, output_file)
        print(f"分词结果已保存到 {output_file}")
    else:
        print("未读取到待分词文本。")
    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"条件随机场耗时: {elapsed_time:.6f}秒")


if __name__ == "__main__":
    main()
