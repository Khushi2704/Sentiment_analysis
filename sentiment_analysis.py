# ===============================
# SENTIMENT ANALYSIS PROJECT
# Amazon Fine Food Reviews
# ===============================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense
)

from wordcloud import WordCloud

# ===============================
# DOWNLOAD STOPWORDS
# ===============================

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ===============================
# LOAD DATASET
# ===============================

print("Loading dataset...")

df = pd.read_csv("E:\Data_science\Sentiment_analysis\Reviews.csv")

print("Original Dataset Shape:", df.shape)

# Optional: Use smaller sample if PC is slow
df = df.sample(50000, random_state=42)

# Keep required columns
df = df[['Text', 'Score']]

df.dropna(inplace=True)

# ===============================
# CREATE SENTIMENT LABEL
# ===============================

df = df[df['Score'] != 3]

df['Sentiment'] = df['Score'].apply(
    lambda x: 1 if x > 3 else 0
)

# ===============================
# TEXT CLEANING FUNCTION
# ===============================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z]',
        ' ',
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

print("Cleaning text...")

df['clean_text'] = df['Text'].apply(
    clean_text
)

# ===============================
# TRAIN TEST SPLIT
# ===============================

X = df['clean_text']

y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# TF-IDF VECTORIZATION
# ===============================

print("Applying TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(
    X_train
)

X_test_tfidf = tfidf.transform(
    X_test
)

# ===============================
# MODEL 1: NAIVE BAYES
# ===============================

print("Training Naive Bayes...")

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_pred = nb_model.predict(
    X_test_tfidf
)

nb_accuracy = accuracy_score(
    y_test,
    nb_pred
)

print("Naive Bayes Accuracy:", nb_accuracy)

# ===============================
# MODEL 2: LOGISTIC REGRESSION
# ===============================

print("Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_pred = lr_model.predict(
    X_test_tfidf
)

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

print("Logistic Regression Accuracy:", lr_accuracy)

# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(
    y_test,
    lr_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix - Logistic Regression")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.close()

# ===============================
# CLASSIFICATION REPORT
# ===============================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        lr_pred
    )
)

# ===============================
# WORDCLOUD
# ===============================

print("Generating WordCloud...")

text = " ".join(
    df['clean_text']
)

wordcloud = WordCloud(
    width=800,
    height=400
).generate(text)

plt.figure(figsize=(10,5))

plt.imshow(wordcloud)

plt.axis('off')

plt.title("WordCloud")

plt.savefig("wordcloud.png")

plt.close()

# ===============================
# SAVE MODELS
# ===============================

print("Saving models...")

pickle.dump(
    lr_model,
    open(
        "sentiment_model.pkl",
        "wb"
    )
)

pickle.dump(
    tfidf,
    open(
        "tfidf_vectorizer.pkl",
        "wb"
    )
)

# ===============================
# LSTM MODEL
# ===============================

print("Preparing LSTM model...")

tokenizer = Tokenizer(
    num_words=5000
)

tokenizer.fit_on_texts(
    X_train
)

X_train_seq = tokenizer.texts_to_sequences(
    X_train
)

X_test_seq = tokenizer.texts_to_sequences(
    X_test
)

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=100
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=100
)

# ===============================
# BUILD LSTM
# ===============================

lstm_model = Sequential()

lstm_model.add(
    Embedding(
        input_dim=5000,
        output_dim=64,
        input_length=100
    )
)

lstm_model.add(
    LSTM(64)
)

lstm_model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

lstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Training LSTM...")

history = lstm_model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# ===============================
# EVALUATE LSTM
# ===============================

lstm_loss, lstm_accuracy = lstm_model.evaluate(
    X_test_pad,
    y_test
)

print("LSTM Accuracy:", lstm_accuracy)

# ===============================
# MODEL COMPARISON
# ===============================

results = pd.DataFrame({

    'Model': [
        'Naive Bayes',
        'Logistic Regression',
        'LSTM'
    ],

    'Accuracy': [
        nb_accuracy,
        lr_accuracy,
        lstm_accuracy
    ]

})

print("\nModel Comparison:\n")

print(results)

# ===============================
# PLOT MODEL COMPARISON
# ===============================

plt.figure(figsize=(8,5))

sns.barplot(
    x='Model',
    y='Accuracy',
    data=results
)

plt.title("Model Comparison")

plt.savefig("model_comparison.png")

plt.close()

print("\nProject Completed Successfully!")