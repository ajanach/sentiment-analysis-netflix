from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

threshold = 0.2 

uri = "mongodb+srv://ajanach:<Password>@cluster0.3rju8fg.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(e)

db_name = "Cluster0" 
db = client[db_name]

collection_name = "tweetovi" 
collection = db[collection_name]

data = list(collection.find())

features = [d['Sadržaj'] for d in data]
sentiments = [float(d['Sentiment']) for d in data]

labels = [1 if sentiment >= threshold else 0 if sentiment <= -threshold else 2 for sentiment in sentiments]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(features)

classifier = RandomForestClassifier(n_estimators=100, max_depth=10)
classifier.fit(X, labels)

model_dir = "trained_model"
os.makedirs(model_dir, exist_ok=True)

model_file = os.path.join(model_dir, "random_forest_model_mongo.pkl")
with open(model_file, 'wb') as file:
    pickle.dump(classifier, file)
    print("Model saved to file:", model_file)

vectorizer_file = os.path.join(model_dir, "vectorizer_random_forest_mongo.pkl")
with open(vectorizer_file, 'wb') as file:
    pickle.dump(vectorizer, file)
    print("Vectorizer saved to file:", vectorizer_file)
