import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

# Definiranje funkcije za čišćenje teksta
def ocisti_tekst(tekst):
    # Zamijeniti tagove korisnika (@) s praznim stringom
    tekst = re.sub(r'@\w+', '', tekst)
    # Zamijeniti hashtagove (#) s praznim stringom
    tekst = re.sub(r'#\w+', '', tekst)
    # Ukloniti sve ostale znakove osim slova i brojeva
    tekst = re.sub("(http\w*:\/)*(\/\w+\.*\w*)+", "", tekst)
    # Vratiti čišćeni tekst
    return tekst

# Čitanje iz CSV datoteke u DataFrame
df = pd.read_csv('twitter_data_100k.csv')

# Primijena funkcije ocisti_tekst() na stupac 'Sadržaj' i spremanje rezultata u novi stupac 'Čišćenje teksta'
df['Čišćenje teksta'] = df['Sadržaj'].apply(ocisti_tekst)


# Requirements for nltk module
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Tokenizacija teksta
df['Tokenizacija'] = df['Čišćenje teksta'].apply(word_tokenize)


# Stvaranje instanci lematizatora i korjenovatelja
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Definicija funkcije za lematizaciju i korjenovanje
def lematizacija_i_korjenovanje(tokens):
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    stems = [stemmer.stem(token) for token in tokens]
    return lemmas, stems

# Primjena lematizacije i korjenovanja na stupac 'Tokenizacija' i spremanje rezultata u nove stupce 'Lematizacija' i 'Korjenovanje'
df['Lematizacija'], df['Korjenovanje'] = zip(*df['Tokenizacija'].apply(lematizacija_i_korjenovanje))


def izbaci_stopword(lista):
    nova_lista = []
    for str in lista:
        if str in stopwords.words('english'):
            continue
        else:
            nova_lista.append(str)
    return nova_lista

df["Zaustavne riječi"] = df["Korjenovanje"].apply(izbaci_stopword)

print(df)

# savim to .csv file
df.to_csv('preprocessing_twitter_data_100k.csv', index=False)
