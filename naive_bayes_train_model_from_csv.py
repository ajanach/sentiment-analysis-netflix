import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Učitavanje modela iz datoteke
model_file = "trained_model\\naive_bayes_model_mongo.pkl"
model_dir = os.path.dirname(model_file)

with open(model_file, 'rb') as file:
    classifier = pickle.load(file)
    print("Model je učitan iz datoteke:", model_file)

# Učitavanje vektorizera iz datoteke
vectorizer_file = os.path.join(model_dir, "vectorizer_mongo.pkl")
with open(vectorizer_file, 'rb') as file:
    vectorizer = pickle.load(file)

# Učitavanje podataka iz .csv datoteke
data_file = "sentiment_analysis_output\\sentiment_analysis_for_testing_3.5k.csv"
data = pd.read_csv(data_file)

# Vektorizacija tekstova
text_vectorized = vectorizer.transform(data['Sadržaj'])

# Predikcija na tekstovima
predictions = classifier.predict(text_vectorized)

# Dodavanje stupca 'Sentiment oznake' s predikcijama u podatke
data['Sentiment oznake NB'] = predictions

# VADER sentiment analiza
analyzer = SentimentIntensityAnalyzer()
sentiments = data['Sadržaj'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Postavljanje praga za klasifikaciju sentimenta
threshold = 0.2  # Prilagodite ovaj prag prema svojim potrebama

# Konvertiranje sentimenta u oznake
labels = [1 if sentiment >= threshold else 0 if sentiment <= -threshold else 2 for sentiment in sentiments]

# Dodavanje stupca 'Sentiment oznake VADER' s oznakama u podatke
data['Sentiment oznake VADER'] = labels

print(data)

# Ispisivanje izvješća o evaluaciji
print("Izvješće o evaluaciji:\n")
print(classification_report(data['Sentiment oznake NB'], data['Sentiment oznake VADER']))

# Ispisivanje matrice zabune
print("Matrica zabune:\n")
print(confusion_matrix(data['Sentiment oznake NB'], data['Sentiment oznake VADER']))

# Ispisivanje točnosti modela
accuracy = accuracy_score(data['Sentiment oznake NB'], data['Sentiment oznake VADER'])
print("Točnost modela:", accuracy)
