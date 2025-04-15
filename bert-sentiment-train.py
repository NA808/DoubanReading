import pandas as pd
import numpy as np
import torch
import gc
import os
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    report = classification_report(labels, preds, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }


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
    return dataset.map(lambda x: x, batched=True, batch_size=64)  # 批量处理


def release_memory():
    """释放 GPU 和 CPU 内存"""
    torch.cuda.empty_cache()
    gc.collect()


def stratified_cross_validation(
        df,
        model_path='/home/roberta-model',
        n_splits=5,
        epochs=5,
        batch_size=16
):
    df['情感标签'] = df['情感标签'].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_f1 = 0
    best_model_path = None
    best_metrics = {}
    tokenizer = BertTokenizer.from_pretrained(model_path)

    for fold, (train_index, val_index) in enumerate(skf.split(df['评论内容'], df['情感标签']), 1):
        print(f"\n===== 训练第 {fold} 折 =====")

        train_texts = df.iloc[train_index]['评论内容'].astype(str).tolist()  # 确保是字符串
        train_labels = df.iloc[train_index]['情感标签'].tolist()
        val_texts = df.iloc[val_index]['评论内容'].astype(str).tolist()  # 确保是字符串
        val_labels = df.iloc[val_index]['情感标签'].tolist()

        train_dataset = prepare_data(train_texts, train_labels, tokenizer)
        val_dataset = prepare_data(val_texts, val_labels, tokenizer)

        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

        training_args = TrainingArguments(
            output_dir=f'/home/results/fold_{fold}',
            save_total_limit=1,  # 只保存最近 1 个 checkpoint
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
            fp16=True,  # 使用混合精度加速
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

        # 记录最佳模型
        f1_score = eval_results['eval_f1']
        if f1_score > best_f1:
            best_f1 = f1_score
            best_model_path = f'/home/best_sentiment_model'
            best_metrics = {
                'accuracy': eval_results['eval_accuracy'],
                'precision': eval_results['eval_precision'],
                'recall': eval_results['eval_recall'],
                'f1': eval_results['eval_f1']
            }

            # 每次找到更好的模型就直接覆盖保存
            trainer.save_model(best_model_path)

        # 释放 GPU 和 CPU 内存
        del model, trainer, train_dataset, val_dataset
        release_memory()

    # 输出最优模型的详细指标
    if best_model_path:
        print("\n🏆 最终最优模型评估结果：")
        print(f"  - 模型保存路径: {best_model_path}")
        print(f"  - 准确率 (Accuracy): {best_metrics['accuracy']:.4f}")
        print(f"  - 精确率 (Precision): {best_metrics['precision']:.4f}")
        print(f"  - 召回率 (Recall): {best_metrics['recall']:.4f}")
        print(f"  - F1 分数 (F1-score): {best_metrics['f1']:.4f}")

    return best_model_path, best_metrics



def main():
    df = pd.read_excel('/home/dataset.xlsx')
    df = df[['评论内容', '情感标签']]
    df.dropna(inplace=True)
    df = df[~df['评论内容'].astype(str).str.isnumeric()]

    stratified_cross_validation(df)


if __name__ == '__main__':
    main()