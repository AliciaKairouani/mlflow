import pandas as pd
from bs4 import BeautifulSoup
# Tokenizer
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
# Télécharger la ressource Punkt
nltk.download('punkt')
nltk.download('wordnet')
# Lemmatizer (base d'un mot)
from nltk.stem import WordNetLemmatizer
# Stop words
from nltk.corpus import stopwords
stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')','`','?']
#fonction pour recuperer les données 
from request import request_api


def remove_all_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    # Extraire uniquement le texte sans balises HTML
    clean_text = soup.get_text(separator=' ', strip=True)
    return clean_text



def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ').replace('(', ' ').replace(')', ' ').replace('`', ' ').replace('?', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw



def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

# Fonction de préparation du texte pour le bag of words avec lemmatization
def transform_bow_lem_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
#    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(word_tokens)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

# Charger les données
data = request_api(score=5,views = 10, filter= 'withbody')
items = data.get('items', [])
# Créez un DataFrame pandas à partir des données extraites
df = pd.DataFrame(items)

def preprocess_data(df):
    # Appliquer la fonction de nettoyage à la colonne spécifique
    df['body'] = df['body'].apply(remove_all_tags)
    df['text'] = df['title'] + ' ' + df['body']
    df['tags'] = df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
    df['sentence_bow'] = df['text'].apply(lambda x : transform_bow_fct(x))
    df['sentence_bow_lem'] = df['text'].apply(lambda x : transform_bow_lem_fct(x))
    df['sentence_dl'] = df['text'].apply(lambda x : transform_dl_fct(x))

    df['length_bow'] = df['sentence_bow'].apply(lambda x : len(word_tokenize(x)))
    #print("max length bow : ", df['length_bow'].max())
    df['length_dl'] = df['sentence_dl'].apply(lambda x : len(word_tokenize(x)))
    #print("max length dl : ", df['length_dl'].max())

    df_final= df[['sentence_dl','sentence_bow_lem','length_bow','length_dl','tags']]
    df_final.rename(columns={'sentence_dl': 'text'},inplace=True, errors='raise')
    return df_final

data = preprocess_data(df)
