from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Preuzimanje resursa za VADER
nltk.download('vader_lexicon')

# Definirajte putanje do modela i vektorizera
model_files = [
    "trained_model/naive_bayes_model_mongo.pkl",
    "trained_model/logistical_regression_model_mongo.pkl",
    "trained_model/random_forest_model_mongo.pkl",
    "trained_model/svm_model_mongo.pkl"
]

vectorizer_files = [
    "trained_model/vectorizer_mongo.pkl",
    "trained_model/vectorizer_logistical_regression_model_mongo.pkl",
    "trained_model/vectorizer_random_forest_mongo.pkl",
    "trained_model/vectorizer_svm_mongo.pkl"
]

# Primjeri novih tekstova za predikciju
new_texts = [
    "@JeremyDewayneH1 @DanielFHart @netflix @wbd This is the image of the perfected Snyderverse. The holy grail right here... So great!  #RestoreTheSnyderVerse #ReleaseTheAyerCut",
    "Awww yeah! @NetflixGeeked @netflix #RENEWTHECUPHEADSHOW",
    "My @Netflix account was stolen for the second time!  Removed my phone and email but still using my Paypal.🤬 Netflix had it resolved in 20 mins. TY!",
    "The show was gr8!",
    "This film sucks, badly!",
    "greedy is a joke app is unusable",
    "Perfect, but badly, badly but perfect.."
]

# Inicijalizacija VADER analizatora sentimenta
sia = SentimentIntensityAnalyzer()

# Prikupljanje rezultata za svaki model
results = []
for model_file, vectorizer_file in zip(model_files, vectorizer_files):
    # Učitavanje modela iz datoteke
    with open(model_file, 'rb') as file:
        classifier = pickle.load(file)
        print("Model loaded from file:", model_file)

    # Učitavanje vektorizera iz datoteke
    with open(vectorizer_file, 'rb') as file:
        vectorizer = pickle.load(file)

    # Vektorizacija novih tekstova
    new_texts_vectorized = vectorizer.transform(new_texts)

    # Predikcija na novim tekstovima
    predictions = classifier.predict(new_texts_vectorized)

    # Evaluacija sentimenta pomoću VADER-a
    true_labels = []  # Stvarne oznake
    for text in new_texts:
        sentiment_scores = sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        if compound_score >= 0.05:
            true_labels.append("Positive")  # Pozitivan sentiment
        elif compound_score <= -0.05:
            true_labels.append("Negative")  # Negativan sentiment
        else:
            true_labels.append("Neutral")  # Neutralan sentiment

    # Rezultati za pojedini model
    model_results = {
        "Text": new_texts,
        "Prediction": predictions,
        "True Label": true_labels
    }
    results.append(model_results)

# Izrada tablica za svaki model
for i, model_file in enumerate(model_files):
    model_name = os.path.basename(model_file).split(".")[0]
    df = pd.DataFrame(results[i])
    print(f"\n--- Evaluacija za model: {model_name} ---")
    pd.set_option('display.max_colwidth', None)  # Prikazivanje cijelog teksta u stupcu
    print(df)
