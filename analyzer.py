import nltk
import string
import random
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import movie_reviews, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download resources (only first run)
nltk.download("movie_reviews")
nltk.download("stopwords")

# ----------------------------
# 1. Collect dataset
# ----------------------------
docs = []

# Take 25 positive + 25 negative reviews
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category)[:25]:
        docs.append((movie_reviews.raw(fileid), category))

# Neutral sentences (custom made)
neutral_headlines = [
    "The weather is expected to be partly cloudy tomorrow",
    "The company announced its quarterly earnings report",
    "A train arrived at the station on schedule",
    "The event will take place at the community center",
    "He walked to the store to buy some groceries",
    "The meeting is scheduled for next Monday",
    "The book was published in 2019 by a local author",
    "The university released its new academic calendar",
    "The temperature is 22 degrees Celsius",
    "The bus stops here every morning at 9 AM",
    "The match ended in a draw with no goals scored",
    "A new bridge was opened for traffic last week",
    "The library has extended its opening hours",
    "The museum will host an art exhibition this weekend",
    "The software update will be available for download soon",
    "Traffic is moving slowly on the main highway",
    "The report was submitted to the committee yesterday",
    "The store opens daily at 10 AM",
    "The office will be closed on public holidays",
    "The lecture will cover modern economic theories"
]

docs.extend([(headline, "neutral") for headline in neutral_headlines])

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
# 3. Feature extraction
# ----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(clean_texts)
y = labels

# ----------------------------
# 4. Train model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ----------------------------
# 5. Predictions
# ----------------------------
preds = model.predict(X)

# ----------------------------
# 6. Results + Visualization
# ----------------------------
df = pd.DataFrame({"Text": texts, "True": labels, "Predicted": preds})
print(df.head(15))

sentiment_counts = df["Predicted"].value_counts()
sentiment_counts.plot(kind="bar", color=["green", "red", "blue"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("chart.png")
plt.show()
