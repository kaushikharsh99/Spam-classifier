import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from collections import Counter

MAX_LEN = 150
MAX_VOCAB = 20000
EMBED_DIM = 128
BATCH_SIZE = 256
EPOCHS = 20

T = 3.0
ALPHA = 0.5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv("df.csv")
df = df.dropna(subset=["text"])
df["text"] = df["text"].str.lower().str.strip()

texts = df["text"].tolist()
labels = df["label"].values

X_train, _, y_train, _ = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=SEED
)

counter = Counter()
for t in X_train:
    counter.update(t.split())

student_vocab = {w: i+1 for i, (w, _) in enumerate(counter.most_common(MAX_VOCAB))}
torch.save(student_vocab, "student_vocab.pt")

teacher_vocab = torch.load("vocab.pt", weights_only=False)

def encode_student(text):
    tokens = [student_vocab.get(w, 0) for w in text.split()]
    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))
    return tokens

def encode_teacher(text):
    tokens = [teacher_vocab.get(w, 0) for w in text.split()]
    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))
    return tokens

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        r = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + r
        x = self.norm(x)
        return x

class Teacher(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 512)
        self.input_fc = nn.Linear(512, 512)
        self.blocks = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
        self.output_fc = nn.Linear(512, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = x.max(dim=1).values
        x = self.input_fc(x)
        x = self.blocks(x)
        return self.output_fc(x)

class Student(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 128)
        self.attn = nn.Linear(128, 1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.embedding(x)
        w = torch.softmax(self.attn(x), dim=1)
        x = (x * w).sum(dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

teacher = Teacher(len(teacher_vocab)).to(device)
teacher.load_state_dict(torch.load("nn_model.pt", map_location=device))
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

student = Student(len(student_vocab)).to(device)

optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

class_counts = torch.bincount(torch.tensor(y_train))
weights = 1. / class_counts.float()

ce_loss = nn.CrossEntropyLoss(weight=weights.to(device))
kl_loss = nn.KLDivLoss(reduction="batchmean")

for epoch in range(EPOCHS):
    student.train()

    for i in range(0, len(X_train), BATCH_SIZE):
        batch_texts = X_train[i:i+BATCH_SIZE]
        batch_labels = y_train[i:i+BATCH_SIZE]

        Xs = torch.tensor([encode_student(t) for t in batch_texts]).to(device)
        Xt = torch.tensor([encode_teacher(t) for t in batch_texts]).to(device)
        y = torch.tensor(batch_labels).to(device)

        with torch.no_grad():
            teacher_logits = teacher(Xt)

        student_logits = student(Xs)

        teacher_soft = torch.softmax(teacher_logits / T, dim=1)
        student_log_soft = torch.log_softmax(student_logits / T, dim=1)

        loss_distill = kl_loss(student_log_soft, teacher_soft) * (T * T)
        loss_ce = ce_loss(student_logits, y)

        loss = ALPHA * loss_ce + (1 - ALPHA) * loss_distill

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

torch.save(student.state_dict(), "student_model.pt")
print("Distillation complete")