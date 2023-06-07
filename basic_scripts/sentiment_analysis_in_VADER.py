import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random

dataset = "datasets\\twitter_data_for_testing_3.5k.csv"
output_file_name = "sentiment_analysis" + str(random.randint(1,1000)) + ".csv"
output_file = "sentiment_analysis_output\\" + output_file_name 

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

    print(df['Sadržaj'], df['Sentiment'])
    print(f"File is saved on path: {output_file}")

sentiment_analysis(dataset, output_file)