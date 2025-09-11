import nltk
import string
import random
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Download resources
nltk.download("movie_reviews")
nltk.download("stopwords")

# ----------------------------
# 1. Collect dataset
# ----------------------------
docs = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category)[:25]:
        docs.append((movie_reviews.raw(fileid), category))

# Neutral examples (limitation: different domain)
neutral_headlines = [
    "The weather is expected to be partly cloudy tomorrow",
    "The company announced its quarterly earnings report",
    "A train arrived at the station on schedule",
    "The event will take place at the community center",
    "He walked to the store to buy some groceries",
]
docs.extend([(h, "neutral") for h in neutral_headlines])

# Shuffle dataset
random.shuffle(docs)
texts, labels = zip(*docs)

# ----------------------------
# 2. Preprocessing
# ----------------------------
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

clean_texts = [preprocess(t) for t in texts]

# ----------------------------
# 3. Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    clean_texts, labels, test_size=0.3, stratify=labels, random_state=42
)

# ----------------------------
# 4. Feature extraction (TF-IDF)
# ----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 5. Train model
# ----------------------------
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# ----------------------------
# 6. Evaluation
# ----------------------------
y_pred = model.predict(X_test_vec)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# 7. Visualization
# ----------------------------
df = pd.DataFrame({"Text": X_test, "True": y_test, "Predicted": y_pred})
sentiment_counts = df["Predicted"].value_counts()
sentiment_counts.plot(kind="bar", color=["green", "red", "blue"])
plt.title("Sentiment Distribution (Test Set)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("chart.png")
plt.show()
