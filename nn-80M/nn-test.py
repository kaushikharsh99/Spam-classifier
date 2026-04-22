import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

MAX_LEN = 150
EMBED_DIM = 512
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

vocab = torch.load("vocab.pt", weights_only=False)
X_test, y_test = torch.load("test_data.pt", weights_only=False)

def encode(text):
    tokens = [vocab.get(w, 0) for w in text.split()]
    tokens = tokens[:MAX_LEN]
    tokens += [0] * (MAX_LEN - len(tokens))
    return tokens

class TestDataset(Dataset):
    def __init__(self, texts, labels):
        self.X = [encode(t) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])

test_loader = DataLoader(
    TestDataset(X_test, y_test),
    batch_size=BATCH_SIZE
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

model.load_state_dict(torch.load("nn_model.pt", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)

        out = model(X)
        preds = out.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total

print("\n===== FINAL TEST RESULT ====")
print(f"Accuracy: {accuracy:.12f}")