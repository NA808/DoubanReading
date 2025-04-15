import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, \
    roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

# 强制使用 Noto Sans CJK SC（简体中文）
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
sns.set(font=font_prop.get_name())
plt.rcParams["axes.unicode_minus"] = False  # 解决负号问题

# 增大字体大小 - 进一步放大字体
SMALL_SIZE = 18  # 从14改为18
MEDIUM_SIZE = 20  # 从16改为20
BIGGER_SIZE = 22  # 从18改为22

plt.rc('font', size=SMALL_SIZE)  # 默认字体大小
plt.rc('axes', titlesize=BIGGER_SIZE)  # 坐标轴标题字体大小
plt.rc('axes', labelsize=MEDIUM_SIZE)  # 坐标轴标签字体大小
plt.rc('xtick', labelsize=SMALL_SIZE)  # x轴刻度标签字体大小
plt.rc('ytick', labelsize=SMALL_SIZE)  # y轴刻度标签字体大小
plt.rc('legend', fontsize=MEDIUM_SIZE)  # 图例字体大小
plt.rc('figure', titlesize=BIGGER_SIZE)  # 图表标题字体大小


# === 读取数据 ===
def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")
    df.dropna(inplace=True)  # 删除缺失值

    # 数据平衡
    df_majority = df[df['情感预测'] == 1]
    df_minority = df[df['情感预测'] == 0]

    # 上采样少数类别
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # 有放回采样
                                     n_samples=len(df_majority),
                                     random_state=42)

    # 合并平衡后的数据集
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    X = df_balanced["评论内容"].astype(str)
    y = df_balanced["情感预测"].astype(int)

    return X, y


# === 高级特征提取 ===
def create_advanced_vectorizer():
    return TfidfVectorizer(
        max_features=10000,  # 进一步增加特征数量
        ngram_range=(1, 3),  # 考虑 1-3 元组
        stop_words=None,  # 可以替换为更专业的停用词表
        max_df=0.9,  # 调整过滤阈值
        min_df=3,  # 调整最小文档频率
        sublinear_tf=True  # 对词频进行对数变换
    )


# === 模型优化流程 ===
def train_and_evaluate_models(X, y):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 高级模型配置
    model_configs = {
        "随机森林": {
            "pipeline": Pipeline([
                ('tfidf', create_advanced_vectorizer()),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            "params": {
                'classifier__n_estimators': [150, 200, 250],
                'classifier__max_depth': [10, 15, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        "支持向量机": {
            "pipeline": Pipeline([
                ('tfidf', create_advanced_vectorizer()),
                ('classifier', LinearSVC(random_state=42))
            ]),
            "params": {
                'classifier__C': [1.0, 1.5, 2.0],
                'classifier__max_iter': [1000, 2000]
            }
        },
        "逻辑回归": {
            "pipeline": Pipeline([
                ('tfidf', create_advanced_vectorizer()),
                ('classifier', LogisticRegression(random_state=42, max_iter=3000))
            ]),
            "params": {
                'classifier__C': [1.0, 1.5, 2.0],
                'classifier__max_iter': [3000, 4000, 5000]
            }
        },
        "朴素贝叶斯": {
            "pipeline": Pipeline([
                ('tfidf', create_advanced_vectorizer()),
                ('classifier', MultinomialNB())
            ]),
            "params": {
                'classifier__alpha': [0.1, 0.5, 1.0, 1.5],
            }
        }
    }

    results = []
    conf_matrices = {}
    classification_reports_dict = {}
    total_metrics = {
        "模型": [],
        "准确率": [],
        "精确率": [],
        "召回率": [],
        "F1值": []
    }

    # 存储最佳模型供ROC曲线使用
    best_models = {}

    for name, config in model_configs.items():
        print(f"正在训练 {name} 模型...")
        # 网格搜索
        grid_search = GridSearchCV(
            config['pipeline'],
            config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # 使用最佳模型
        best_model = grid_search.best_estimator_
        best_models[name] = best_model  # 保存最佳模型
        y_pred = best_model.predict(X_test)

        # 评估指标
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # 存储结果
        results.append([name, acc, precision, recall, f1])

        # 记录总体指标
        total_metrics["模型"].append(name)
        total_metrics["准确率"].append(acc)
        total_metrics["精确率"].append(precision)
        total_metrics["召回率"].append(recall)
        total_metrics["F1值"].append(f1)

        # 分类报告
        report = classification_report(y_test, y_pred, digits=4)
        classification_reports_dict[name] = report
        print(f"{name} 训练完成，最佳参数: {grid_search.best_params_}")
        print(f"准确率: {acc:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        conf_matrices[name] = cm

    # 返回结果，现在包括最佳模型字典
    return best_models, X_test, y_test, results, classification_reports_dict, conf_matrices


# === 绘制 ROC 曲线 ===
def plot_roc_curve(X_test, y_test, best_models):
    plt.figure(figsize=(12, 10))

    # 定义不同的线型和颜色
    linestyles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']

    i = 0
    # 为每个支持predict_proba的模型绘制ROC曲线
    for name, model in best_models.items():
        try:
            # 检查模型是否有predict_proba方法
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr,
                         label=f'{name} (AUC = {roc_auc:.2f})',
                         linestyle=linestyles[i % len(linestyles)],
                         color=colors[i % len(colors)],
                         linewidth=3)
            # 对于LinearSVC，我们可以使用decision_function作为替代
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr,
                         label=f'{name} (AUC = {roc_auc:.2f})',
                         linestyle=linestyles[i % len(linestyles)],
                         color=colors[i % len(colors)],
                         linewidth=3)
            else:
                print(f"模型 {name} 不支持概率预测或决策函数，跳过ROC曲线")
            i += 1
        except Exception as e:
            print(f"绘制 {name} 的ROC曲线时出错: {e}")

    # 设置图表属性 - 增大字体
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=20, fontweight='bold')
    plt.ylabel('真正率 (TPR)', fontsize=20, fontweight='bold')
    plt.title('不同模型的 ROC 曲线', fontsize=24, fontweight='bold')
    plt.legend(loc='lower right', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 增大刻度标签字体
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# 主执行流程
def main():
    print("开始情感分析模型评估...")

    # 加载数据
    try:
        X, y = load_and_preprocess_data("predicted_sentiments.xlsx")
        print(f"数据加载完成，共 {len(X)} 条评论")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 训练和评估模型
    try:
        best_models, X_test, y_test, results, classification_reports, conf_matrices = train_and_evaluate_models(X, y)
    except Exception as e:
        print(f"模型训练失败: {e}")
        return

    # === 保存分类报告 ===
    try:
        with open("classification_reports.txt", "w", encoding="utf-8") as f:
            for name, report in classification_reports.items():
                f.write(f"{name} 分类报告:\n{report}\n\n")
        print("分类报告已保存至 classification_reports.txt")
    except Exception as e:
        print(f"保存分类报告失败: {e}")

    # === 可视化 ===
    try:
        # 混淆矩阵 - 增大字体和数字
        plt.figure(figsize=(16, 13))
        for i, (name, cm) in enumerate(conf_matrices.items(), 1):
            plt.subplot(2, 2, i)
            sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', linewidths=0.5,
                        xticklabels=["负面", "正面"],
                        yticklabels=["负面", "正面"],
                        annot_kws={"size": 20})  # 增大数字大小
            plt.xlabel("预测值", fontsize=20, fontweight='bold')
            plt.ylabel("真实值", fontsize=20, fontweight='bold')
            plt.title(f"{name} - 混淆矩阵", fontsize=22, fontweight='bold')

            # 增大刻度标签
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)

        plt.tight_layout()
        plt.savefig("confusion_matrices.png", dpi=300, bbox_inches='tight')
        print("混淆矩阵图片已保存至 confusion_matrices.png")

        # 模型性能对比 - 使用明显不同的颜色，并去掉底部的"模型"标签
        result_df = pd.DataFrame(results, columns=["模型", "准确率", "精确率", "召回率", "F1 值"])
        plt.figure(figsize=(12, 8))

        # 定义对比度更高的颜色方案
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # 更新的barplot语法
        bars = sns.barplot(x="模型", y="准确率", data=result_df.sort_values(by="准确率", ascending=False),
                           palette=distinct_colors, legend=False)

        # 添加数值标签到柱形图顶部，增大字体
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.005,
                f'{bar.get_height():.4f}',
                ha='center',
                fontsize=18  # 增大数字标签大小
            )

        # 不显示底部的"模型"标签
        plt.xlabel("", fontsize=20, fontweight='bold')  # 修改这里，去掉"模型"标签
        plt.ylabel("准确率", fontsize=20, fontweight='bold')
        plt.title("不同模型的准确率对比", fontsize=24, fontweight='bold')
        plt.xticks(rotation=45, fontsize=18, fontweight='bold')
        plt.yticks(fontsize=18)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches='tight')
        print("准确率对比图已保存至 accuracy_comparison.png")

        # 绘制 ROC 曲线
        try:
            plot_roc_curve(X_test, y_test, best_models)
            print("ROC曲线图已保存至 roc_curve_comparison.png")
        except Exception as e:
            print(f"绘制ROC曲线失败: {e}")

        # 保存结果
        result_df.to_excel("model_results.xlsx", index=False)
        print("结果数据表已保存至 model_results.xlsx")
    except Exception as e:
        print(f"可视化或保存结果失败: {e}")

    print("\n✅ 所有完成的结果：")
    print("- 混淆矩阵图片: confusion_matrices.png")
    print("- 准确率对比图: accuracy_comparison.png")
    print("- ROC曲线图: roc_curve_comparison.png")
    print("- 分类报告: classification_reports.txt")
    print("- 结果数据表: model_results.xlsx")


if __name__ == "__main__":
    main()