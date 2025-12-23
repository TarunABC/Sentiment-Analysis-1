import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

from preprocess import clean_text
from model import get_model

# Load data
data = pd.read_csv("data/train.tsv", sep="\t")

data["sentence"] = data["sentence"].apply(clean_text)

X = data["sentence"]
y = data["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = get_model()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

dump(model, "sentiment_model.joblib")
dump(vectorizer, "tfidf_vectorizer.joblib")
