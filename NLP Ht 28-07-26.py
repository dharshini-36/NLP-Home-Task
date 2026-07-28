import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("sentiment_analysis.csv")
print(data.head())

# Download Stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text Preprocessing
def preprocess(text):
    text = str(text).lower()

    # Remove numbers and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and apply stemming
    words = [stemmer.stem(word)
             for word in words
             if word not in stop_words]

    return " ".join(words)

# Apply preprocessing
data["clean_text"] = data["text"].apply(preprocess)

print(data[["text", "clean_text"]].head())

# Feature Extraction using TF-IDF
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(data["clean_text"])
y = data["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test Model
prediction = model.predict(X_test)

print("Predictions:")
print(prediction)

print("\nAccuracy:")
print(accuracy_score(y_test, prediction))

print("\nClassification Report:")
print(classification_report(y_test, prediction))

# Get sentence from user
user_sentence = input("Enter a sentence: ")

# Preprocess the sentence
clean_sentence = preprocess(user_sentence)

# Convert into TF-IDF
new_text = tfidf.transform([clean_sentence])

# Predict sentiment
result = model.predict(new_text)

print("Predicted Sentiment:", result[0])
