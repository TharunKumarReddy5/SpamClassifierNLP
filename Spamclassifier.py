# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 23:14:43 2021

@author: Tharun Kumar Reddy Karasani
"""

import pandas as pd
# importing the Dataset
messages = pd.read_csv('smsspamcollection\SMSSpamCollection', sep='\t', names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
wl = WordNetLemmatizer()

def preprocess(convert_type='stem'):
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
        review = review.lower()
        review = review.split()
        if convert_type=='stem':
            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        else:
            review = [wl.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

# Creating the Bag of Words and TF-IDF model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cv = CountVectorizer(max_features=2500)
tf = TfidfVectorizer(max_features=2500)

# Encoding target labels using one hot encoding and avoiding dummy variable trap
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values
    
def train_predict(corpus, transform_type="bow"):
    if transform_type=='bow':
        X = cv.fit_transform(corpus).toarray()
    else:
        X = tf.fit_transform(corpus).toarray()

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Training model using Naive bayes classifier
    spam_detect_model = MultinomialNB().fit(X_train, y_train)
    y_pred = spam_detect_model.predict(X_test)
    # Evaluation Metrics
    print(confusion_matrix(y_test,y_pred))
    print("The accuracy score is {:0.2f}".format(accuracy_score(y_test,y_pred)))
    return y_test,y_pred

train_predict(preprocess("stem"),"bow")
train_predict(preprocess("stem"),"tfidf")
train_predict(preprocess("lem"),"bow")
train_predict(preprocess("lem"),"tfidf")