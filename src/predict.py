from joblib import load
from preprocess import clean_text

model = load("sentiment_model.joblib")
vectorizer = load("tfidf_vectorizer.joblib")

def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

if __name__ == "__main__":
    text = input("Enter a sentence: ")
    print("Sentiment:", predict_sentiment(text))
