import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")


fake_df["label"] = "Fake"
true_df["label"] = "True"


df = pd.concat([fake_df, true_df], ignore_index=True)


df = df[["text", "label"]]


X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("===== Fake News Detection using Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_articles = [
    "The government announced a new education policy to improve higher education.",
    "Scientists discovered a new planet that is made entirely of gold.",
    "The Reserve Bank increased interest rates to control inflation.",
    "Aliens have officially signed a peace treaty with world leaders.",
    "Researchers developed a vaccine that passed all clinical trials successfully."
]

new_articles_tfidf = tfidf.transform(new_articles)

predictions = model.predict(new_articles_tfidf)

print("\n===== New Article Predictions =====")

for article, prediction in zip(new_articles, predictions):
    print(f"{prediction} --> {article}")