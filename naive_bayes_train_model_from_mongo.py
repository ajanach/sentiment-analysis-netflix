from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
import pickle
import os

# Postavke pragova za klasifikaciju sentimenta
threshold = 0.2  # Prilagodite ovaj prag prema svojim potrebama

# Postavite svoje podatke o spajanju na bazu
uri = "mongodb+srv://ajanach:Pa$$w0rd@cluster0.3rju8fg.mongodb.net/?retryWrites=true&w=majority"

# Stvaranje novog klijenta i spajanje na server
client = MongoClient(uri, server_api=ServerApi('1'))

# Slanje pinga za potvrdu uspješne veze
try:
    client.admin.command('ping')
    print("Uspješno spajanje na MongoDB!")
except Exception as e:
    print(e)

# Spajanje na bazu
db_name = "Cluster0"  # Zamijeniti s imenom baze podataka

# Stvaranje referencu na bazu
db = client[db_name]

# Definiranje kolekcije iz koje želite povući tweetove
collection_name = "tweetovi"  # Zamijenite s imenom kolekcije

# Stvaranje referencu na kolekciju
collection = db[collection_name]

# Povlačenje podataka iz baze
data = list(collection.find())

# Separacija podataka u značajke i oznake
features = [d['Sadržaj'] for d in data]
sentiments = [float(d['Sentiment']) for d in data]

# Konvertiranje sentimenta u oznake
labels = [1 if sentiment >= threshold else 0 if sentiment <= -threshold else 2 for sentiment in sentiments]

# Vektorizacija značajki
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(features)

# Treniranje Naivnog Bayesovog klasifikatora
classifier = MultinomialNB()
classifier.fit(X, labels)

# Spremanje modela i vectorizera u datoteke
model_dir = "trained_model"
os.makedirs(model_dir, exist_ok=True)

model_file = os.path.join(model_dir, "naive_bayes_model_mongo.pkl")
with open(model_file, 'wb') as file:
    pickle.dump(classifier, file)
    print("Model je spremljen u datoteku:", model_file)

vectorizer_file = os.path.join(model_dir, "vectorizer_mongo.pkl")
with open(vectorizer_file, 'wb') as file:
    pickle.dump(vectorizer, file)
    print("Vectorizer je spremljen u datoteku:", vectorizer_file)
