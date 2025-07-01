import pandas as pd
import re
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 加载停用词表
def load_stopwords(stopwords_file="hit_stopwords.txt"):
    with open(stopwords_file, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f.readlines()])

# 文本清洗 + 分词 + 停用词去除
def clean_and_tokenize(text, stopwords):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)  # 仅保留中文字符
    words = jieba.lcut(text)
    words = [word for word in words if word not in stopwords]
    return " ".join(words)

# 生成词云图
def generate_wordcloud(text, title, filename):
    wc = WordCloud(font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 中文字体路径
                   background_color="white",
                   width=800,
                   height=600,
                   max_words=200).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"{title} 已保存为 {filename}")

# 主函数：读取Excel并绘制词云
def main():
    # 加载数据
    df = pd.read_excel("dataset.xlsx", usecols=["评论内容", "情感标签"])

    # 加载停用词
    stopwords = load_stopwords("hit_stopwords.txt")

    # 清洗和分词
    df["评论内容"] = df["评论内容"].apply(lambda x: clean_and_tokenize(x, stopwords))

    # 根据情感标签分组
    pos_text = " ".join(df[df["情感标签"] == 1]["评论内容"])
    neg_text = " ".join(df[df["情感标签"] == 0]["评论内容"])

    # 生成词云图
    generate_wordcloud(pos_text, "正向评论词云图", "positive_wordcloud.png")
    generate_wordcloud(neg_text, "负向评论词云图", "negative_wordcloud.png")

if __name__ == "__main__":
    main()
