import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 设置中文字体
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [my_font.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# === 加载数据 ===
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")
    df.dropna(inplace=True)
    print("标签分布：\n", df['情感标签'].value_counts())
    return df["评论内容"].astype(str), df["情感标签"].astype(int)

# === 向量器 ===
def create_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words=None,
        max_df=0.9,
        min_df=3,
        sublinear_tf=True
    )

# === 模型训练与绘图 ===
def train_and_plot(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "朴素贝叶斯": {
            "pipeline": Pipeline([
                ('tfidf', create_vectorizer()),
                ('clf', MultinomialNB())
            ]),
            "params": {
                'clf__alpha': [0.5, 1.0]
            }
        },
        "支持向量机": {
            "pipeline": Pipeline([
                ('tfidf', create_vectorizer()),
                ('clf', LinearSVC(random_state=42))
            ]),
            "params": {
                'clf__C': [1.0, 2.0],
                'clf__max_iter': [1000]
            }
        }
    }

    for name, config in models.items():
        print(f"\n训练模型：{name}")
        gs = GridSearchCV(config["pipeline"], config["params"], cv=5, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print(f"最优参数: {gs.best_params_}")
        print(f"准确率: {acc:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")

        # ==== 混淆矩阵 ====
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                         xticklabels=['负面', '正面'],
                         yticklabels=['负面', '正面'],
                         annot_kws={"size": 18})
        ax.set_xticklabels(['负面', '正面'], fontsize=18, fontproperties=my_font)
        ax.set_yticklabels(['负面', '正面'], fontsize=18, fontproperties=my_font, rotation=0)
        plt.xlabel('预测结果', fontsize=20, fontproperties=my_font)
        plt.ylabel('真实情况', fontsize=20, fontproperties=my_font)
        plt.title(f'{name} - 混淆矩阵', fontsize=22, fontproperties=my_font)
        plt.tight_layout()
        plt.savefig(f"{name}_confusion_matrix.png", dpi=600, bbox_inches='tight')
        plt.show()

        # ==== ROC 曲线 ====
        if hasattr(best_model.named_steps['clf'], "decision_function"):
            scores = best_model.decision_function(X_test)
        else:
            scores = best_model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=3, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5)
        plt.xlabel('假阳性率', fontsize=20, fontproperties=my_font)
        plt.ylabel('真阳性率', fontsize=20, fontproperties=my_font)
        plt.title(f'{name} - ROC 曲线', fontsize=22, fontproperties=my_font)
        plt.legend(loc='lower right', fontsize=18, prop=my_font)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f"{name}_roc_curve.png", dpi=600, bbox_inches='tight')
        plt.show()

# === 主程序 ===
def main():
    X, y = load_and_preprocess_data("cleaned_dataset.xlsx")
    print(f"加载完成，共 {len(X)} 条评论")
    train_and_plot(X, y)

if __name__ == "__main__":
    main()

