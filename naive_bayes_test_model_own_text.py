from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, accuracy_score
from sklearn.metrics import PrecisionRecallDisplay

# Preuzimanje resursa za VADER
nltk.download('vader_lexicon')

# Učitavanje modela iz datoteke
model_file = "trained_model/naive_bayes_model_mongo.pkl"
model_dir = os.path.dirname(model_file)

with open(model_file, 'rb') as file:
    classifier = pickle.load(file)
    print("Model je učitan iz datoteke:", model_file)

# Primjeri novih tekstova za predikciju
new_texts = [
    "@JeremyDewayneH1 @DanielFHart @netflix @wbd This is the image of the perfected Snyderverse. The holy grail right here...  #RestoreTheSnyderVerse #ReleaseTheAyerCut",
    "Awww yeah! @NetflixGeeked @netflix #RENEWTHECUPHEADSHOW",
    "My @Netflix account was stolen for the second time!  Removed my phone and email but still using my Paypal.🤬 Netflix had it resolved in 20 mins. TY!",
    "The show was gr8!",
    "This film sucks, badly!"
]

# Vektorizacija novih tekstova
vectorizer_file = os.path.join(model_dir, "vectorizer_mongo.pkl")
with open(vectorizer_file, 'rb') as file:
    vectorizer = pickle.load(file)

new_texts_vectorized = vectorizer.transform(new_texts)

# Predikcija na novim tekstovima
predictions = classifier.predict(new_texts_vectorized)

# Inicijalizacija VADER analizatora sentimenta
sia = SentimentIntensityAnalyzer()

# Prikaz predikcija i VADER analize
true_labels = []  # Stvarne oznake

for text in new_texts:
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        true_labels.append(1)  # Pozitivan sentiment
    elif compound_score <= -0.05:
        true_labels.append(0)  # Negativan sentiment
    else:
        true_labels.append(2)  # Neutralan sentiment

for text, prediction, true_label in zip(new_texts, predictions, true_labels):
    if prediction == 1:
        sentiment_label = "Pozitivan"
    elif prediction == 0:
        sentiment_label = "Negativan"
    else:
        sentiment_label = "Neutralan"

    print("Tekst:", text)
    print("Predikcija Naivnog Bayes modela:", prediction)
    print("Stvarna oznaka sentimenta (VADER):", true_label)
    print("Sentiment:", sentiment_label)
    print()
