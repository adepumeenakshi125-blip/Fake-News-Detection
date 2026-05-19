import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load datasets
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# Add labels
fake_df["label"] = "FAKE"
true_df["label"] = "REAL"

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0)

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Remove null values
df = df.fillna("")

# Create content column
df["content"] = df["text"]

# Features and labels
X = df["content"]
y = df["label"]

# Convert labels
# FAKE = 0
# REAL = 1
y = y.map({"FAKE": 0, "REAL": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)

model.fit(X_train_vectorized, y_train)

# Prediction
y_pred = model.predict(X_test_vectorized)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and Vectorizer saved successfully!")