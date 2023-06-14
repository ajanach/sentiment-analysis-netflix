import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model_files = [
    {
        "model_file": "trained_model/logistical_regression_model_mongo.pkl",
        "vectorizer_file": "trained_model/vectorizer_logistical_regression_model_mongo.pkl",
        "model_name": "Logistical Regression"
    },
    {
        "model_file": "trained_model/naive_bayes_model_mongo.pkl",
        "vectorizer_file": "trained_model/vectorizer_mongo.pkl",
        "model_name": "Naive Bayes (MongoDB)"
    },
    {
        "model_file": "trained_model/random_forest_model_mongo.pkl",
        "vectorizer_file": "trained_model/vectorizer_random_forest_mongo.pkl",
        "model_name": "Random Forest"
    },
    {
        "model_file": "trained_model/svm_model_mongo.pkl",
        "vectorizer_file": "trained_model/vectorizer_svm_mongo.pkl",
        "model_name": "SVM"
    }
]

for model_info in model_files:
    model_file = model_info["model_file"]
    vectorizer_file = model_info["vectorizer_file"]
    model_name = model_info["model_name"]

    with open(model_file, 'rb') as file:
        classifier = pickle.load(file)
        print("Model loaded from file:", model_file)

    with open(vectorizer_file, 'rb') as file:
        vectorizer = pickle.load(file)

    data_file = "sentiment_analysis_output/sentiment_analysis_for_testing_3.5k.csv"
    data = pd.read_csv(data_file)

    text_vectorized = vectorizer.transform(data['Sadržaj'])
    predictions = classifier.predict(text_vectorized)
    data['Sentiment oznake Model'] = predictions

    analyzer = SentimentIntensityAnalyzer()
    sentiments = data['Sadržaj'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    threshold = 0.2
    labels = [1 if sentiment >= threshold else 0 if sentiment <= -threshold else 2 for sentiment in sentiments]
    data['Sentiment oznake VADER'] = labels

    print("Evaluation Report for", model_name, ":\n")
    print(classification_report(data['Sentiment oznake Model'], data['Sentiment oznake VADER']))
    print("Confusion Matrix:\n")
    print(confusion_matrix(data['Sentiment oznake Model'], data['Sentiment oznake VADER']))
    accuracy = accuracy_score(data['Sentiment oznake Model'], data['Sentiment oznake VADER'])
    print("Model Accuracy:", accuracy)

    nb_sentiment_counts = data['Sentiment oznake Model'].value_counts()
    nb_labels = ['Positive', 'Negative', 'Neutral']
    nb_sizes = nb_sentiment_counts.values
    nb_colors = ['#ff9999', '#66b3ff', '#99ff99']

    plt.figure(figsize=(6, 6))
    plt.pie(nb_sizes, labels=nb_labels, colors=nb_colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Sentiment Distribution (' + model_name + ')')
    plt.show()

    vader_sentiment_counts = data['Sentiment oznake VADER'].value_counts()
    vader_labels = ['Positive', 'Negative', 'Neutral']
    vader_sizes = vader_sentiment_counts.values
    vader_colors = ['#ff9999', '#66b3ff', '#99ff99']

    plt.figure(figsize=(6, 6))
    plt.pie(vader_sizes, labels=vader_labels, colors=vader_colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')
    plt.title('Sentiment Distribution (VADER)')
    plt.show()
