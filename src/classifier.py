from typing import List

import torch

from sklearn.preprocessing import LabelEncoder

from support import *

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





class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """
    def __init__(self):

        self.label_encoder = LabelEncoder()
        self.max_len = 0
        self.features = None
        self.labels = None
     


    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        dataset = data_clean(train_filename)
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'princeton-nlp/unsup-simcse-bert-large-uncased')
        
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights).to(torch.device(device))

        features = tokenize(self.tokenizer,self.model, dataset,device)
        dataset['integer_sentiment'] = self.label_encoder.fit_transform(dataset.sentiment)
        labels = dataset['integer_sentiment']

        self.clf = SVC(gamma='auto')
        self.clf.fit(features,labels)

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        dataset = data_clean(data_filename)
        features = tokenize(self.tokenizer,self.model, dataset,device)

        predictions = self.clf.predict(features)
        return self.label_encoder.inverse_transform(predictions)



