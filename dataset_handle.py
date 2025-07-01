import pandas as pd
import re
import jieba

# 加载停用词表（哈工大停用词）
def load_stopwords(stopwords_file="hit_stopwords.txt"):
    with open(stopwords_file, "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f.readlines()])
    return stopwords

# 文本清洗 + 分词 + 停用词去除
def clean_and_tokenize(text, stopwords):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)  # 保留中文
    words = jieba.lcut(text)
    words = [word for word in words if word not in stopwords]
    return " ".join(words)

def process_single_file(input_file, output_file, stopwords_file="hit_stopwords.txt"):
    stopwords = load_stopwords(stopwords_file)

    # 读取评论和情感列（假设情感标签列名为“情感”）
    df = pd.read_excel(input_file, usecols=['评论内容', '情感标签'])
    original_count = len(df)

    # 清洗评论内容
    df['评论内容'] = df['评论内容'].apply(lambda x: clean_and_tokenize(x, stopwords))

    # 去重、去空
    df.drop_duplicates(subset=['评论内容'], inplace=True)
    df.dropna(subset=['评论内容', '情感标签'], inplace=True)
    cleaned_count = len(df)

    # 保存清洗后的数据
    df.to_excel(output_file, index=False)

    print(f"原始评论数量: {original_count} 条")
    print(f"处理后有效评论数量: {cleaned_count} 条")

    # 统计每个情感标签的数量
    label_counts = df['情感标签'].value_counts()
    print("\n各情感标签数量：")
    for label, count in label_counts.items():
        print(f"{label}: {count} 条")

    print(f"\n数据处理完成，已保存至: {output_file}")

def main():
    input_file = 'dataset.xlsx'
    output_file = 'cleaned_dataset.xlsx'
    process_single_file(input_file, output_file)

if __name__ == '__main__':
    main()
