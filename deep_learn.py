import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from tqdm import tqdm

# 字体设置（替换为你的系统路径）
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [my_font.get_name()]
plt.rcParams['axes.unicode_minus'] = False

# 超参数
MAX_LEN = 128
BATCH_SIZE = 32
#EPOCHS = 30
EPOCHS = 20
NUM_CLASSES = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
df = pd.read_excel("cleaned_dataset.xlsx")
df = df[df['情感标签'].isin([0, 1])]
df['评论内容'] = df['评论内容'].astype(str)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['评论内容'].tolist(),
    df['情感标签'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['情感标签']
)

# Tokenizer & 模型
tokenizer = BertTokenizer.from_pretrained("/home/roberta-model")
bert_model = BertModel.from_pretrained("/home/roberta-model").to(device)

# 数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(TextDataset(train_texts, train_labels), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TextDataset(test_texts, test_labels), batch_size=BATCH_SIZE)

# BiLSTM模型
'''class RobertaBiLSTM(nn.Module):
    def __init__(self, bert, hidden_dim=256, num_classes=2, dropout=0.3):
        super(RobertaBiLSTM, self).__init__()
        self.bert = bert
        self.lstm = nn.LSTM(768, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)'''
# 5. TextCNN 模型
class RobertaTextCNN(nn.Module):
    def __init__(self, bert, num_classes):
        super(RobertaTextCNN, self).__init__()
        self.bert = bert
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, 768)) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100 * 3, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # 冻结BERT/RoBERTa参数
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.unsqueeze(1)  # [B, 1, L, D]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(B,100,Lk),...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]   # [(B,100),...]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 评估函数
def evaluate(model, dataloader):
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            preds.extend(predictions.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            probs.extend(probabilities[:, 1].cpu().tolist())  # 正类概率
    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
    return acc, precision, recall, f1, preds, targets, probs

# 绘图函数（无PR曲线）
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=['负面', '正面'],
                     yticklabels=['负面', '正面'],
                     annot_kws={"size": 18})
    ax.set_xticklabels(['负面', '正面'], fontsize=18, fontproperties=my_font)
    ax.set_yticklabels(['负面', '正面'], fontsize=18, fontproperties=my_font, rotation=0)
    plt.xlabel('预测结果', fontsize=20, fontproperties=my_font)
    plt.ylabel('真实情况', fontsize=20, fontproperties=my_font)
    plt.title('TextCNN-混淆矩阵', fontsize=22, fontproperties=my_font)
    plt.tight_layout()
    plt.savefig("confusion_matrix_TextCNN.png", dpi=600, bbox_inches="tight")
    plt.show()

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=3, label=f'曲线下面积 AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1.5)
    plt.xlabel('假阳性率', fontsize=20, fontproperties=my_font)
    plt.ylabel('真阳性率', fontsize=20, fontproperties=my_font)
    plt.title('TextCNN-ROC 曲线', fontsize=22, fontproperties=my_font)
    plt.legend(loc='lower right', fontsize=18, prop=my_font)
    plt.grid()
    plt.tight_layout()
    plt.savefig("roc_curve_TextCNN.png", dpi=600, bbox_inches="tight")
    plt.show()

# 模型训练与评估
#model = RobertaBiLSTM(bert_model, hidden_dim=256, num_classes=NUM_CLASSES).to(device)
model = RobertaTextCNN(bert_model, NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    acc, precision, recall, f1, preds, targets, probs = evaluate(model, test_loader)
    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# 绘图（无PR曲线）
plot_confusion_matrix(targets, preds)
plot_roc_curve(targets, probs)
