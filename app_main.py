import streamlit as st
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
from clean import preprocess_predict

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

tag_df = pd.read_csv("tag.csv")
def get_related_tags(predoutput):
    
    tagfilter = tag_df[tag_df['Predicted_Tag'] == predoutput]
    return tagfilter["tags"].sample(1).value

# Title and Subtitle
st.title("Tags Detector")

# Input box for the word
input_word = st.text_input("Enter your word")

# Button to trigger related tags
if st.button("Get Related Tags"):
    if input_word:
        # Displaying related tags
        st.subheader("Tags Related")
        input_word = [input_word]
        imput =  pd.DataFrame(input_word, columns=['MaColonne'])
        # st.title(model.predict(imput)[0])
        related_tags = get_related_tags(model.predict(imput)[0])
        st.write(related_tags)
    else:
        st.warning("Please enter a word to get related tags.")
