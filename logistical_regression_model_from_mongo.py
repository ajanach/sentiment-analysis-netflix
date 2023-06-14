from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
import pickle
import os

# Postavke pragova za klasifikaciju sentimenta
treshold = 0.2

# Postaviti svoje podatke o spajanju na bazu
uri = "mongodb+srv://ajanach:<Password>@cluster0.3rju8fg.mongodb.net/?retryWrites=true&w=majority"

# Stvaranje novog klijenta i spajanje na server
client = MongoClient(uri, server_api=ServerApi('1'))

# Slanje pinga za potvrdu uspješne veze
try:
    client.admin.command('ping')
    print("Uspješno spojen na MongoDB!")
except Exception as e:
    print(e)

# Spajanje na bazu
naziv_baze = "Cluster0" 
baza = client[naziv_baze]

# Definiranje kolekcije iz koje želite povući tweetove
naziv_kolekcije = "tweetovi" 
kolekcija = baza[naziv_kolekcije]

# Povlačenje podataka iz baze
data = list(kolekcija.find())

# Separacija podataka u značajke i oznake
features = [d['Sadržaj'] for d in data]
sentiment = [float(d['Sentiment']) for d in data]

# Konvertiranje osjećaja u oznake
oznake = [1 if osjećaj >= treshold else 0 if osjećaj <= -treshold else 2 for osjećaj in sentiment]

# Vektorizacija značajki
vektorizator = CountVectorizer()
X = vektorizator.fit_transform(features)

# Treniranje logističke regresije
klasifikator = LogisticRegression()
klasifikator = LogisticRegression(max_iter=1000)
klasifikator.fit(X, oznake)

# Spremanje modela i vektorizatora u datoteke
putanja_modela = "trained_model"
os.makedirs(putanja_modela, exist_ok=True)

datoteka_modela = os.path.join(putanja_modela, "logisticka_regresija_model_mongo.pkl")
with open(datoteka_modela, 'wb') as datoteka:
    pickle.dump(klasifikator, datoteka)
    print("Model je spremljen u datoteku:", datoteka_modela)

datoteka_vektorizatora = os.path.join(putanja_modela, "vectorizer_logisticka_regresija_model_mongo.pkl")
with open(datoteka_vektorizatora, 'wb') as datoteka:
    pickle.dump(vektorizator, datoteka)
    print("Vektorizator je spremljen u datoteku:", datoteka_vektorizatora)
