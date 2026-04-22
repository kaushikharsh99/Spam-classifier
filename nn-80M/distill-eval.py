import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MAX_LEN = 150
EMBED_DIM = 128
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

student_vocab = torch.load("student_vocab.pt", weights_only=False)
X_test, y_test = torch.load("test_data.pt", weights_only=False)

def encode(text):
    tokens = [student_vocab.get(w, 0) for w in text.split()]
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

class Student(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, EMBED_DIM)
        self.attn = nn.Linear(EMBED_DIM, 1)
        self.fc1 = nn.Linear(EMBED_DIM, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.embedding(x)
        w = torch.softmax(self.attn(x), dim=1)
        x = (x * w).sum(dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Student(len(student_vocab)).to(device)
model.load_state_dict(torch.load("student_model.pt", map_location=device))
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

print("\n===== SMALL STUDENT MODEL RESULT =====")
print(f"Accuracy: {accuracy:.12f}")