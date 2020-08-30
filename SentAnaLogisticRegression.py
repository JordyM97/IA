import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
import numpy as np

Dataset = 'covid-data-clases.csv'
df = pd.read_csv(Dataset, names=['tweets', 'sentimientos'], sep=';')


tweets = df['tweets'].values



y = df['sentimientos'].values
tweets_train, tweets_test, y_train, y_test = train_test_split(tweets, y, test_size=0.20, random_state=100)
vectorizer = CountVectorizer()
vectorizer.fit(tweets_train)
X_train = vectorizer.transform(tweets_train)
X_test = vectorizer.transform(tweets_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Accuracy:", score)
test = vectorizer.transform(['el dia de hoy no hay infectados'])
print(classifier.predict(test))





