import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification


def predict_sentiment(texts, model, tokenizer, device, batch_size=16):
    """
    批量进行情感预测

    参数:
    - texts: 待预测的文本列表
    - model: 预训练的BERT模型
    - tokenizer: 文本编码器
    - device: 计算设备(CPU/GPU)
    - batch_size: 每批次处理的文本数量

    返回:
    预测的情感标签列表
    """
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # 文本编码
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)

    return predictions


def main():
    # 设备检测：如果支持 GPU + FP16，则使用 FP16，否则使用 FP32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 模型和分词器路径
    model_path = "/home/best_sentiment_model"

    print(f"🔍 开始加载模型和分词器...")

    # 加载 Tokenizer 和训练好的 BERT 模型
    tokenizer = BertTokenizer.from_pretrained("/home/roberta-model")
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,  # 二分类
        torch_dtype=torch_dtype
    ).to(device)

    # 进入推理模式，减少显存占用
    model.eval()

    # 读取未标注数据
    input_file = "/home/combined_comments.xlsx"
    print(f"📖 读取输入文件: {input_file}")

    df = pd.read_excel(input_file)

    # 删除空值
    df.dropna(subset=['评论内容'], inplace=True)

    # 过滤掉空字符串
    df = df[df['评论内容'].str.strip() != '']

    texts = df["评论内容"].tolist()

    print(f"📊 准备预测 {len(texts)} 条评论...")

    # 进行情感预测
    df["情感标签"] = predict_sentiment(
        texts,
        model,
        tokenizer,
        device,
        batch_size=16
    )

    # 保存结果（移除 encoding 参数）
    output_file = "/home/predicted_sentiments.xlsx"
    df.to_excel(output_file, index=False)

    print(f"✅ 预测完成，结果已保存至 {output_file}")
    print(f"   总评论数: {len(texts)}")
    print(f"   正向情感: {sum(df['情感标签'] == 1)}")
    print(f"   负向情感: {sum(df['情感标签'] == 0)}")


if __name__ == "__main__":
    main()