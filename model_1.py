import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
import spacy
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
from request import request_api
from clean import preprocess_data
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import pickle

#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.start_run() 

data = request_api(score=5,views = 10, filter= 'withbody')
items = data.get('items', [])
# Créez un DataFrame pandas à partir des données extraites
df = pd.DataFrame(items)

# Prétraitement des données
preprocessed_data = preprocess_data(df)
preprocessed_data 

data = preprocessed_data[['text', 'tags']]
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['tags'], test_size=0.2, random_state=42)


def classify_tags(df, word2vec_model, num_clusters=3):
    # Tokeniser le document en mots
    df['tokens'] = df['text'].apply(lambda x: x.split())

    # Vectoriser chaque document en utilisant le modèle Word2Vec
    df['vectors'] = df['tokens'].apply(lambda tokens: [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv])

    # Agréger les vecteurs pour chaque document
    df['avg_vector'] = df['vectors'].apply(lambda vectors: sum(vectors) / len(vectors) if vectors else None)

    # Supprimer les lignes avec des vecteurs nuls
    df = df.dropna(subset=['avg_vector'])

    # Transformer les vecteurs en un tableau 2D pour KMeans
    X = pd.DataFrame(df['avg_vector'].tolist())

    # Appliquer le regroupement K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    # Mapper les clusters à des tags en utilisant LabelEncoder
    label_encoder = LabelEncoder()
    df['Predicted_Tag'] = label_encoder.fit_transform(df['Cluster'])

    # Retourner le DataFrame avec les prédictions
    return df[['text', 'tags', 'Predicted_Tag']]



# Remplacez 'votre_dataframe.csv' par le chemin de votre fichier CSV Stack Overflow
word2vec_model = Word2Vec(sentences=data, vector_size=500, window=5, min_count=1, workers=4)

# Appliquer la fonction de classification
result_df = classify_tags(data, word2vec_model)

# Afficher les résultats
print(result_df)

X=result_df['text']
y=result_df['Predicted_Tag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

# Serialize (pickle) the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(nb, file)