import csv

file = "datasets\\twitter_data_for_testing_3.5k.csv"

# Provjeri broj zapisanih tweetova
def check_number_of_tweets(output_file):
    with open(output_file, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        num_written_tweets = len(list(reader)) - 1  # Oduzmi zaglavlje

    print(f"Broj zapisanih tweetova: {num_written_tweets}")

check_number_of_tweets(file)