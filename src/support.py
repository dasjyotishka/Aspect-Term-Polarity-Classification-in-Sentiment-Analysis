from typing import List

import torch

from sklearn.preprocessing import LabelEncoder


import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import transformers as ppb
import warnings
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize        
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


    #import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def data_clean(dataset_path):
  dataset = pd.read_csv(dataset_path, sep='\t', header= None)
  # print(dataset)
  dataset = dataset.loc[:, [0, 1, 2,3, 4]]
  dataset = dataset.rename(index=str, columns={ 0: "sentiment", 1: "aspect_category",2: "word", 3:"position", 4: "review"})
  label_encoder = LabelEncoder()
  
  dataset['review'] = dataset['review'].apply(str.lower)
  dataset['word'] = dataset['word'].apply(str.lower)

# We try to keep all the no/nor/not words as this changes radically the sentiment analysis
  dataset['review'] = dataset["review"].apply(lambda sentence: sentence.replace("can\'t", "can not"))
  dataset['review'] = dataset["review"].apply(lambda sentence: sentence.replace("n\'t", " not"))
  return dataset

def add_aspect(review, aspect):
    return '[ASPECT=' + aspect + '] ' + review
    
def tokenize(tokenizer,model, dataset,device):
    dataset['review_as'] = dataset.apply(lambda row: add_aspect(row['review'], row['aspect_category']), axis=1)
    tokenized = dataset['review_as'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    max_len = max([len(i) for i in tokenized.values])
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids.to(torch.device(device)), attention_mask=attention_mask.to(torch.device(device)))
    features = last_hidden_states[0][:,0,:].cpu().numpy()
    
    # Load stopwords and stemmer from NLTK
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.remove('nor')
    stopwords.remove('no')
    stopwords.remove('not')
    stemmer = nltk.stem.PorterStemmer()

    #Define a custom tokenizer that uses the stemmer
    def tokenizer_l(text):
        return [stemmer.stem(word) for word in nltk.word_tokenize(text.lower()) if word not in stopwords]
    # Create a count vectorizer with custom tokenizer
    count_vectorizer = CountVectorizer(tokenizer=tokenizer_l)
    # Fit the count vectorizer on the reviews
    count_features = count_vectorizer.fit_transform(dataset['review']).toarray()
    # Create a Latent Dirichlet Allocation (LDA) model with 10 topics
    lda = LatentDirichletAllocation(n_components=10)

    # Fit the LDA model on the count features
    lda_features = lda.fit_transform(count_features)


    return np.concatenate([features, lda_features], axis=1)