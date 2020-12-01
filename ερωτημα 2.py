import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import  precision_score, recall_score, f1_score
  
df = pd.read_csv(r'C:\Users\User\Desktop\project εξορυξη\onion-or-not.csv', sep = ',')
y=df.label

#tokenization
df['tokenized_sents'] = df.apply(lambda column: nltk.word_tokenize(column['text']), axis=1) 
#tokens = df['tokenized_sents']

ps = PorterStemmer() 
#stemming

df['stemmed'] = df['tokenized_sents'].apply(lambda x: [ps.stem(y) for y in x])
stemmed = df['stemmed']

stemmed_ = pd.DataFrame(stemmed) 

#stopwords removal
stop_words = set(stopwords.words('english')) 

stemmed.apply(lambda x: [item for item in x if item not in stop_words])


tfidf_vectorizer = TfidfVectorizer()
dataframe = stemmed.to_frame()

dataframe['stemmed']=[" ".join(review) for review in df['stemmed'].values] 
tfidf = tfidf_vectorizer.fit_transform(dataframe['stemmed'])  

#print(tfidf_vectorizer.vocabulary_)
    

x_train, x_test, y_train, y_test = train_test_split(stemmed, y,  test_size = 0.25)

model = Sequential()
model.add(Dense(12,input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train,epochs=20, batch_size=1, verbose=1)
y_pred = model.predict(x_test)
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test,y_pred))
