import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
import numpy as np
import matplotlib.pyplot as plt
from plot_keras_history import plot_history

class Classifier:
    dataset="covid-data-clases.csv"
    classifier=LogisticRegression()
    score=0
    vectorizer = CountVectorizer()
    def __init__(self):
        df = pd.read_csv(self.dataset, names=['tweets', 'sentimientos'], sep=';')
        tweets = df['tweets'].values

        y = df['sentimientos'].values
        tweets_train, tweets_test, y_train, y_test = train_test_split(tweets, y, test_size=0.20, random_state=20)
        
        self.vectorizer.fit(tweets_train)
        X_train = self.vectorizer.transform(tweets_train)
        X_test = self.vectorizer.transform(tweets_test)

        self.classifier.fit(X_train, y_train)
        self.score = self.classifier.score(X_test, y_test)
        










