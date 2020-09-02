import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
import numpy as np
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
from sklearn.feature_extraction import stop_words
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
class Classifier:
    dataset="coviddatasetfinal.csv"
    classifier=LogisticRegression()
    score=0
    X_train=0
    def __init__(self):
        stop=stop_words.ENGLISH_STOP_WORDS
        df = pd.read_csv(self.dataset, names=['tweets', 'sentimientos'], sep=',')    
        st= pd.read_csv("stopwords", sep='\n', header=None)
        stop=stop.union(list(st[0]))
        y = df['sentimientos'].values
        tweets = df['tweets'].values
        tweets_train, tweets_test, y_train, y_test = train_test_split(tweets, y, test_size=0.10, random_state=20)
        self.vectorizer=CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=8, max_df=0.8, stop_words=stop)
        self.vectorizer.fit(tweets_train)
        self.X_train = self.vectorizer.transform(tweets_train)
        self.X_test = self.vectorizer.transform(tweets_test)

        self.classifier.fit(self.X_train, y_train)
        self.score = self.classifier.score(self.X_test, y_test)
        result= self.classifier.predict(self.X_test)
        print(classification_report(y_test,result))











