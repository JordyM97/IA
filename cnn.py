import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
import numpy as np
from keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

Dataset = 'db2.csv'
df = pd.read_csv(Dataset, names=['sentence', 'label'], sep='|')


tweet = df.sentence.values
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

y = df['label'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(padded_sequence, y,test_size=0.20, random_state=100)
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length,input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model.fit(sentences_train, y_train, epochs=6, batch_size=32)
score = model.evaluate(sentences_test, y_test)
print("Certeza:", score[1])

test_word ="Esto es muy malo, detesto esto"
tw = tokenizer.texts_to_sequences([test_word])
tw = pad_sequences(tw,maxlen=200)
prediction = int(model.predict(tw).round().item())
print(prediction)