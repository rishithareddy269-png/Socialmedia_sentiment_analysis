import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# 1. Load combined dataset
# ==============================

data_path = "data/combined_social_media.csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==============================
# 2. Remove missing values
# ==============================

df = df.dropna(subset=["text", "sentiment"])

df["text"] = df["text"].astype(str)


# ==============================
# 3. Prepare X and y
# ==============================

X = df["text"]
y = df["sentiment"]


# ==============================
# 4. Split dataset
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 5. TF-IDF Vectorization
# ==============================

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF conversion completed!")


# ==============================
# 6. Train ML model
# ==============================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# ==============================
# 7. Make predictions
# ==============================

y_pred = model.predict(X_test_tfidf)


# ==============================
# 8. Evaluate model
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==============================
# 9. Create model folder
# ==============================

os.makedirs("model", exist_ok=True)


# ==============================
# 10. Save model
# ==============================

joblib.dump(
    model,
    "model/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


print("\n==============================")
print("FILES SAVED SUCCESSFULLY")
print("==============================")

print("model/sentiment_model.pkl")
print("model/tfidf_vectorizer.pkl")