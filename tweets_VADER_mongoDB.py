import snscrape.modules.twitter as sntwitter
import csv
from datetime import datetime, timedelta
import time
from sentiment_anlysis_VADER import sentiment_analysis
import pymongo
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Definiranje filtera za prikupljanje tweetova
filter = "@netflix -filter:retweets lang:en"  # Dodaj "lang:en" za filtriranje engleskih tweetova

# Postavljanje broja tweetova koje želimo prikupiti
num_tweets = 100000

# Postavljanje izlazne .csv datoteke - za dataset
output_file_dataset = "datasets\\dataset.csv"

# Postavljanje vremenskog raspona
start_date = datetime(2023, 5, 1).date()
end_date = datetime(2023, 6, 1).date()

# Mjeri vrijeme izvršavanja
start_time = time.time()

# Otvaranje datoteke za pisanje
with open(output_file_dataset, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ID", "Datum", "Autor", "Sadržaj"])

    # Prikupljanje tweetova i pisanje u .csv datoteku
    for tweet in sntwitter.TwitterSearchScraper(filter + f" since:{start_date} until:{end_date}").get_items():
        writer.writerow([tweet.id, tweet.date, tweet.user.username, tweet.rawContent])
        num_tweets -= 1

        if num_tweets == 0:
            break

# Izračunaj proteklo vrijeme
elapsed_time = time.time() - start_time

print("Prikupljanje podataka s Twittera je završeno.")
print(f"Izvršavanje vremena: {elapsed_time:.2f} sekundi.")

# Sentiment Analysis - VADER
out_sentiment_analysis = "sentiment_analysis_output\\sentiment_analysis.csv"
sentiment_analysis("datasets\\twitter_data_100k.csv", out_sentiment_analysis)

# Postaviti svoje podatke o spajanju na bazu
uri = "mongodb+srv://ajanach:<Password>@cluster0.3rju8fg.mongodb.net/?retryWrites=true&w=majority"

# Stvaranje novog klijenta i spajanje na server
client = MongoClient(uri, server_api=ServerApi('1'))

# Slanje pinga za potvrdu uspješne veze
try:
    client.admin.command('ping')
    print("Uspješno spajanje na MongoDB!")
except Exception as e:
    print(e)

# Spajanje na bazu
db_name = "Cluster0"  # Zamjeniti s imenom baze podataka
db = client[db_name]

# Definiranje kolekcije u koju želitm spremiti tweetove
collection_name = "tweetovi"  # zamjeniti s imenom kolekcije

# Provjera zadnjeg datuma preuzetih tweetova iz kolekcije
last_tweet = db[collection_name].find().sort("Datum", pymongo.DESCENDING).limit(1)
last_tweet = list(last_tweet)
last_date = last_tweet[0]["Datum"] if last_tweet else datetime.min

# Prikupljanje novih tweetova i spremanje u bazu
with open(out_sentiment_analysis, "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Preskoči zaglavlje
    for row in reader:
        tweet_date = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S+00:00")
        if tweet_date.date() > last_date.date():
            tweet = {
                "ID": row[0],
                "Datum": tweet_date,
                "Autor": row[2],
                "Sadržaj": row[3],
                "Sentiment": row[4]
            }
            db[collection_name].insert_one(tweet)

print("Spremanje novih tweetova u MongoDB bazu podataka je završeno.")
tweet_count = db[collection_name].count_documents({})
print("Broj tweetova u kolekciji:", tweet_count)
