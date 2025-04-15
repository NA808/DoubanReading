import pandas as pd
import os
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


    # 只保留中文字符
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)

    # 分词
    words = jieba.lcut(text)

    # 去除停用词
    words = [word for word in words if word not in stopwords]

    # 重新拼接
    return " ".join(words)

def process_comments(directory, output_file, stopwords_file="hit_stopwords.txt"):
    stopwords = load_stopwords(stopwords_file)
    all_comments = []

    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            df = pd.read_excel(file_path, usecols=['评论内容'])
            all_comments.append(df)

    all_comments_df = pd.concat(all_comments, ignore_index=True)

    # 清洗 + 分词 + 去停用词
    all_comments_df['评论内容'] = all_comments_df['评论内容'].apply(lambda x: clean_and_tokenize(x, stopwords))

    # 去重 & 去空值
    all_comments_df.drop_duplicates(subset=['评论内容'], inplace=True)
    all_comments_df.dropna(subset=['评论内容'], inplace=True)

    # 保存处理后的数据
    all_comments_df.to_excel(output_file, index=False)

    print(f"数据处理完成！共 {len(all_comments_df)} 条有效评论，已保存至 {output_file}")
    return all_comments_df

def main():
    directory = 'douban_comments'  # 你的评论数据文件夹
    output_file = 'combined_comments.xlsx'
    process_comments(directory, output_file)

if __name__ == '__main__':
    main()
