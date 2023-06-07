import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import matplotlib.pyplot as plt

# dataset = "Datasets\\twitter_data_100k.csv"
# output_file_name = "sentiment_analysis" + str(random.randint(1,1000)) + ".csv"
# output_file = "sentiment_analysis_output\\" + output_file_name 

def sentiment_analysis(dataset, output_file):
    # Reguirements za VADER leksikon
    nltk.download('vader_lexicon')

    # Učitavanje .csv datoteke u DataFrame
    df = pd.read_csv(dataset)

    # Inicijalizacija VADER sentiment analizatora
    sia = SentimentIntensityAnalyzer()

    # Primjena VADER analize sentimenta na stupac "Sadržaj"
    df['Sentiment'] = df['Sadržaj'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Spremanje rezultata analize sentimenta u novu .csv datoteku
    df.to_csv(output_file, index=False)

    # print(df['Sadržaj'], df['Sentiment'])
    print(f"File is saved on path: {output_file}")

# Učitavanje podataka iz .csv datoteke
data_file = "sentiment_analysis_output\\sentiment_analysis_100k_dataset.csv"
data = pd.read_csv(data_file)

# Postavljanje praga za klasifikaciju sentimenta
threshold = 0.2  # Prilagodite ovaj prag prema svojim potrebama

# Analiza sentimenta koristeći VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
data['Sentiment_VADER'] = data['Sadržaj'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Konvertiranje sentimenta u oznake
labels = [1 if sentiment >= threshold else 0 if sentiment <= -threshold else 2 for sentiment in data['Sentiment_VADER']]

# Dodavanje stupca 'Sentiment oznake VADER' s oznakama u podatke
data['Sentiment oznake VADER'] = labels

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