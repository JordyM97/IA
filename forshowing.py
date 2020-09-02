import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import stop_words
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import CategoricalNB
import numpy as np


stop=stop_words.ENGLISH_STOP_WORDS
print(list(stop))

Dataset = 'coviddatasetfinal.csv'
df = pd.read_csv(Dataset, names=['tweets', 'sentimientos'], sep=',')
st= pd.read_csv("stopwords", sep='\n', header=None)
stop=stop.union(list(st[0]))

tweets = df['tweets'].values
val = df['sentimientos'].values
print(list(val).count(1))

y = df['sentimientos'].values
tweets_train, tweets_test, y_train, y_test = train_test_split(tweets, y, test_size=0.20, random_state=0)
vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=8, max_df=0.8, stop_words=stop)
vectorizer.fit(tweets_train)
X_train = vectorizer.transform(tweets_train)
X_test = vectorizer.transform(tweets_test)
print(X_train.shape)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("Con una precision del",score)
test = vectorizer.transform(['Disfrutar'])
result= classifier.predict(X_test)

print(result)


#print(accuracy_score(y_test, result))
#print(confusion_matrix(y_test,result))
print(classification_report(y_test,result))

if result[0] == 1:
    print("Positivo")
elif result[0] == 0:
    print("Neutro")
else:
    print("Negativo")