import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

start_time = time.time()
print("Loading data...")
df = pd.read_csv("df.csv")

df = df.dropna(subset=["text"])
df['text'] = df['text'].str.lower().str.strip()

print("Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)


print("Vectorizing text (this may take time)...")
vec_start = time.time()

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

vec_end = time.time()
print(f"Vectorization done in {vec_end - vec_start:.2f} seconds")

print("Training model...")
train_start = time.time()

model = LogisticRegression(max_iter=20000, n_jobs=-1)
model.fit(X_train_vec, y_train)

train_end = time.time()
print(f"Training done in {train_end - train_start:.2f} seconds")

print("\n===== TRAIN PERFORMANCE =====")
y_train_pred = model.predict(X_train_vec)
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

print("\n===== TEST PERFORMANCE =====")
y_test_pred = model.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

end_time = time.time()
print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")

print("\nModel and vectorizer saved ")