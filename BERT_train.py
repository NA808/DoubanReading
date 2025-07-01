import pandas as pd
import numpy as np
import torch
import gc
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.font_manager as fm


def compute_metrics(pred):
    """只计算指标，不绘制图片"""
    labels = pred.label_ids
    probs = pred.predictions
    preds = probs.argmax(-1)

    report = classification_report(labels, preds, output_dict=True)
    conf_matrix = confusion_matrix(labels, preds)

    # 计算 SN（灵敏度/敏感性）和 SP（特异性）
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 计算AUC但不绘图
    auc_score = 0.0
    try:
        if probs.shape[1] == 2:
            prob_positive = probs[:, 1]
            auc_score = roc_auc_score(labels, prob_positive)
    except Exception as e:
        print(f"⚠️ AUC 计算失败：{e}")

    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': conf_matrix.tolist(),
        'auc': auc_score
    }


def plot_final_results(trainer, fold_num):
    """只在最后一折绘制图片"""
    # 在验证集上进行预测
    eval_results = trainer.predict(trainer.eval_dataset)
    labels = eval_results.label_ids
    probs = eval_results.predictions
    preds = probs.argmax(-1)

    conf_matrix = confusion_matrix(labels, preds)

    # 指定中文字体路径
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    my_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [my_font.get_name()]
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                     xticklabels=['负面', '正面'],
                     yticklabels=['负面', '正面'],
                     annot_kws={"size": 18})
    ax.set_xticklabels(['负面', '正面'], fontsize=18, fontproperties=my_font)
    ax.set_yticklabels(['负面', '正面'], fontsize=18, fontproperties=my_font, rotation=0)
    plt.xlabel('预测结果', fontsize=20, fontproperties=my_font)
    plt.ylabel('真实情况', fontsize=20, fontproperties=my_font)
    plt.title('BERT-混淆矩阵', fontsize=22, fontproperties=my_font)
    plt.tight_layout()
    plt.savefig("confusion_matrix_BERT.png", dpi=600, bbox_inches="tight")
    plt.show()

    # 绘制 ROC AUC 曲线
    try:
        if probs.shape[1] == 2:
            prob_positive = probs[:, 1]
            auc_score = roc_auc_score(labels, prob_positive)
            fpr, tpr, _ = roc_curve(labels, prob_positive)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'曲线下面积 AUC = {auc_score:.2f}')
            plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5)
            plt.xlabel('假阳性率', fontsize=20, fontproperties=my_font)
            plt.ylabel('真阳性率', fontsize=20, fontproperties=my_font)
            plt.title('BERT-ROC 曲线', fontsize=22, fontproperties=my_font)
            plt.legend(loc='lower right', fontsize=18, prop=my_font)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.grid(False)
            plt.tight_layout()
            plt.savefig("roc_curve_BERT.png", dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"⚠️ AUC 绘图失败：{e}")


def prepare_data(texts, labels, tokenizer, max_length=512):
    """优化 Dataset 处理，减少内存占用"""
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length
    )
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    return dataset.map(lambda x: x, batched=True, batch_size=64)


def release_memory():
    """释放 GPU 和 CPU 内存"""
    torch.cuda.empty_cache()
    gc.collect()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_cross_validation(
        df,
        model_path='/home/roberta-model',
        n_splits=5,
        epochs=5,
        batch_size=16
):
    df['情感标签'] = df['情感标签'].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # 用于存储每一折的指标
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'sensitivity': [],
        'specificity': []
    }

    for fold, (train_index, val_index) in enumerate(skf.split(df['评论内容'], df['情感标签']), 1):
        print(f"\n===== 训练第 {fold} 折 =====")

        train_texts = df.iloc[train_index]['评论内容'].astype(str).tolist()
        train_labels = df.iloc[train_index]['情感标签'].tolist()
        val_texts = df.iloc[val_index]['评论内容'].astype(str).tolist()
        val_labels = df.iloc[val_index]['情感标签'].tolist()

        train_dataset = prepare_data(train_texts, train_labels, tokenizer)
        val_dataset = prepare_data(val_texts, val_labels, tokenizer)

        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

        training_args = TrainingArguments(
            output_dir=f'/home/results/fold_{fold}',
            save_total_limit=1,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            logging_dir=f'/home/logs/fold_{fold}',
            logging_steps=10,
            fp16=True,
            report_to=["none"]
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        eval_results = trainer.evaluate()

        # 收集每一折的指标
        fold_metrics['accuracy'].append(eval_results['eval_accuracy'])
        fold_metrics['precision'].append(eval_results['eval_precision'])
        fold_metrics['recall'].append(eval_results['eval_recall'])
        fold_metrics['f1'].append(eval_results['eval_f1'])
        fold_metrics['sensitivity'].append(eval_results['eval_sensitivity'])
        fold_metrics['specificity'].append(eval_results['eval_specificity'])

        # 输出当前折的指标
        print(f"第 {fold} 折的准确率: {eval_results['eval_accuracy']:.4f}")
        print(f"第 {fold} 折的精确率: {eval_results['eval_precision']:.4f}")
        print(f"第 {fold} 折的召回率: {eval_results['eval_recall']:.4f}")
        print(f"第 {fold} 折的 F1 分数: {eval_results['eval_f1']:.4f}")
        print(f"第 {fold} 折的敏感性 (SN): {eval_results['eval_sensitivity']:.4f}")
        print(f"第 {fold} 折的特异性 (SP): {eval_results['eval_specificity']:.4f}")

        # 只在最后一折绘制图片
        if fold == n_splits:
            print(f"\n===== 绘制第 {fold} 折的图片 =====")
            plot_final_results(trainer, fold)

        # 释放 GPU 和 CPU 内存
        del model, trainer, train_dataset, val_dataset
        release_memory()

    # 输出所有折的准确率
    print("\n每一折的准确率如下：")
    for i, acc in enumerate(fold_metrics['accuracy'], 1):
        print(f"第 {i} 折准确率: {acc:.4f}")

    # 计算并输出平均指标
    print("\n===== 五折交叉验证平均结果 =====")
    avg_accuracy = np.mean(fold_metrics['accuracy'])
    avg_precision = np.mean(fold_metrics['precision'])
    avg_recall = np.mean(fold_metrics['recall'])
    avg_f1 = np.mean(fold_metrics['f1'])
    avg_sensitivity = np.mean(fold_metrics['sensitivity'])
    avg_specificity = np.mean(fold_metrics['specificity'])

    print(f"平均准确率 (Accuracy): {avg_accuracy:.4f}")
    print(f"平均精确率 (Precision): {avg_precision:.4f}")
    print(f"平均召回率 (Recall): {avg_recall:.4f}")
    print(f"平均 F1 分数 (F1-score): {avg_f1:.4f}")
    print(f"平均敏感性 (SN): {avg_sensitivity:.4f}")
    print(f"平均特异性 (SP): {avg_specificity:.4f}")

    return {
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_sensitivity': avg_sensitivity,
        'avg_specificity': avg_specificity
    }


def main():
    df = pd.read_excel('/home/dataset.xlsx')
    df = df[['评论内容', '情感标签']]
    df.dropna(inplace=True)
    df = df[~df['评论内容'].astype(str).str.isnumeric()]

    set_seed(42)
    stratified_cross_validation(df)


if __name__ == '__main__':
    main()