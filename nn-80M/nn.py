import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import time
import os

MAX_VOCAB = 80000
MAX_LEN = 150
EMBED_DIM = 512
BATCH_SIZE = 256
EPOCHS = 30
LOG_EVERY = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv("df.csv")
df = df.dropna(subset=["text"])
df['text'] = df['text'].str.lower().str.strip()

texts = df['text'].tolist()
labels = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

torch.save((X_test, y_test), "test_data.pt")

counter = Counter()
for text in X_train:
    counter.update(text.split())

vocab = {w: i+1 for i, (w, _) in enumerate(counter.most_common(MAX_VOCAB))}
torch.save(vocab, "vocab.pt")

def encode(text):
    tokens = [vocab.get(w, 0) for w in text.split()]
    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))
    return tokens

class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.X = [encode(t) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

train_loader = DataLoader(
    SpamDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        r = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + r
        x = self.norm(x)
        return x

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, EMBED_DIM)
        self.input_fc = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.blocks = nn.Sequential(
            ResidualBlock(EMBED_DIM),
            ResidualBlock(EMBED_DIM),
            ResidualBlock(EMBED_DIM),
            ResidualBlock(EMBED_DIM)
        )
        self.output_fc = nn.Linear(EMBED_DIM, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = x.max(dim=1).values
        x = self.input_fc(x)
        x = self.blocks(x)
        return self.output_fc(x)

model = Model(len(vocab)).to(device)

class_counts = torch.bincount(torch.tensor(y_train))
weights = 1. / class_counts.float()
loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

log_path = "losslogs.txt"
file_exists = os.path.exists(log_path)
log_file = open(log_path, "a")

if not file_exists:
    log_file.write("step,epoch,loss,eta\n")

global_step = 0

for epoch in range(EPOCHS):
    model.train()
    start_time = time.time()

    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        global_step += 1

        if global_step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / len(train_loader)
            eta = elapsed / progress - elapsed

            log_line = f"{global_step},{epoch+1},{loss.item():.6f},{eta:.2f}\n"
            log_file.write(log_line)
            log_file.flush()

            print(f"Step {global_step} | Loss {loss.item():.4f} | ETA {eta:.1f}s")

torch.save(model.state_dict(), "nn_model.pt")

log_file.close()

print("Training complete")