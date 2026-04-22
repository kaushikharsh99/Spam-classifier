import pandas as pd
import joblib
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


print(f"Using {os.cpu_count()} CPU cores")

start_total = time.time()

print("\nLoading data...")
df = pd.read_csv("df.csv")

df = df.dropna(subset=["text"])
df['text'] = df['text'].str.lower().str.strip()

texts = df['text'].tolist()
labels = df['label'].values

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print("Vectorizing...")
t0 = time.time()

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    stop_words='english',
    dtype='float32'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vectorization time: {time.time() - t0:.2f}s")

print("Training SVM...")
t1 = time.time()

model = LinearSVC(
    class_weight='balanced',
    max_iter=2000
)

model.fit(X_train_vec, y_train)

print(f"Training time: {time.time() - t1:.2f}s")


print("\n===== TRAIN =====")
y_train_pred = model.predict(X_train_vec)
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))

print("\n===== TEST =====")
y_test_pred = model.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

joblib.dump(model, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nSaved model + vectorizer")

print(f"\nTotal runtime: {time.time() - start_total:.2f}s")