from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import os

# Postavke pragova za klasifikaciju sentimenta
threshold = 0.2

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
db_name = "Cluster0"

# Stvaranje referencu na bazu
db = client[db_name]

# Definiranje kolekcije iz koje želite povući tweetove
collection_name = "tweetovi"

# Stvaranje referencu na kolekciju
collection = db[collection_name]

# Povlačenje podataka iz baze
cursor = collection.find({}, {"Sadržaj": 1, "Sentiment": 1})
data = list(cursor)

# Separacija podataka u značajke i oznake
features = [d['Sadržaj'] for d in data]
sentiments = [float(d['Sentiment']) for d in data]

# Konvertiranje sentimenta u oznake
labels = [1 if sentiment >= threshold else 0 if sentiment <= -threshold else 2 for sentiment in sentiments]

# Vektorizacija značajki
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(features)

# Treniranje SVM klasifikatora
classifier = LinearSVC()
classifier.fit(X, labels)

# Spremanje modela i vektorizatora u datoteke
model_dir = "trained_model"
os.makedirs(model_dir, exist_ok=True)

model_file = os.path.join(model_dir, "svm_model_mongo.pkl")
with open(model_file, 'wb') as file:
    pickle.dump(classifier, file)
    print("Model je spremljen u datoteku:", model_file)

vectorizer_file = os.path.join(model_dir, "vectorizer_svm_mongo.pkl")
with open(vectorizer_file, 'wb') as file:
    pickle.dump(vectorizer, file)
    print("Vektorizator je spremljen u datoteku:", vectorizer_file)
