# install snscrape: "pip3 install git+https://github.com/JustAnotherArchivist/snscrape.git"

import snscrape.modules.twitter as sntwitter
import csv
from datetime import datetime, timedelta
import time

# Definiranje filtera za prikupljanje tweetova
filter = "@netflix -filter:retweets lang:en"  # Dodaj "lang:en" za filtriranje engleskih tweetova

# Postavljanje izlazne .csv datoteke
output_file = "datasets\\twitter_data_for_testing_3.5k.csv"

# Postavljanje broja tweetova koje želimo prikupiti
num_tweets = 3050

# Postavljanje vremenskog raspona
start_date = datetime(2023, 5, 1)
end_date = datetime(2023, 6, 2)

# Mjeri vrijeme izvršavanja
start_time = time.time()

# Otvaranje datoteke za pisanje
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ID", "Datum", "Autor", "Sadržaj"])

    # Prikupljanje tweetova i pisanje u .csv datoteku
    for tweet in sntwitter.TwitterSearchScraper(filter + f" since:{start_date.strftime('%Y-%m-%d')} until:{end_date.strftime('%Y-%m-%d')}").get_items():
        writer.writerow([tweet.id, tweet.date, tweet.user.username, tweet.content])
        num_tweets -= 1

        if num_tweets == 0:
            break

# Izračunaj proteklo vrijeme
elapsed_time = time.time() - start_time

print("Prikupljanje podataka s Twittera je završeno.")
print(f"Izvršavanje vremena: {elapsed_time:.2f} sekundi.")