import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# Dodavanje stupca 'Sentiment oznake NB' s predikcijama Naive Bayes modela u podatke
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

# Ispisivanje izvješća o evaluaciji
print("Izvješće o evaluaciji:\n")
print(classification_report(data['Sentiment oznake NB'], data['Sentiment oznake VADER']))

# Ispisivanje matrice zabune
print("Matrica zabune:\n")
print(confusion_matrix(data['Sentiment oznake NB'], data['Sentiment oznake VADER']))

# Ispisivanje točnosti modela
accuracy = accuracy_score(data['Sentiment oznake NB'], data['Sentiment oznake VADER'])
print("Točnost modela:", accuracy)

# Kreiranje pie charta za Naive Bayes predikcije
nb_sentiment_counts = data['Sentiment oznake NB'].value_counts()
nb_labels = ['Pozitivan', 'Negativan', 'Neutralan']
nb_sizes = nb_sentiment_counts.values
nb_colors = ['#ff9999', '#66b3ff', '#99ff99']  # Opcionalno: Dodajte boje

plt.figure(figsize=(6, 6))
plt.pie(nb_sizes, labels=nb_labels, colors=nb_colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # Omogućite da je pie chart okrugao
plt.title('Raspodjela sentimenta (Naive Bayes)')
plt.show()

# Kreiranje pie charta za analizu sentimenta VADER
vader_sentiment_counts = data['Sentiment oznake VADER'].value_counts()
vader_labels = ['Pozitivan', 'Negativan', 'Neutralan']
vader_sizes = vader_sentiment_counts.values
vader_colors = ['#ff9999', '#66b3ff', '#99ff99']  # Opcionalno: Dodajte boje

plt.figure(figsize=(6, 6))
plt.pie(vader_sizes, labels=vader_labels, colors=vader_colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # Omogućite da je pie chart okrugao
plt.title('Raspodjela sentimenta (VADER)')
plt.show()
