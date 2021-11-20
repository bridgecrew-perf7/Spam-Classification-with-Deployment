# Basic imports 

from os import remove
import pandas as pd
import numpy 
import sklearn
import nltk
import re

# Few imports of nltk and sklearn


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# Data import and preprocessing

dataset=pd.read_csv(r'C:\Users\20065\OneDrive\Documents\NLP-Python\spam.csv',encoding='latin-1')

X=dataset['v2']

y=dataset['v1']

y=pd.get_dummies(y,drop_first=True)


# Data cleaning

stopwords=stopwords.words('english')

lemmatizer=WordNetLemmatizer()

n,=X.shape


for i in range(n):
    temp=re.sub('[^a-zA-Z]',' ',X[i])
    temp=temp.lower()
    temp=temp.split()
    temp=[lemmatizer.lemmatize(word) for word in temp if word not in set(stopwords)]
    temp=' '.join(temp)
    X[i]=temp




cv=CountVectorizer()

X=cv.fit_transform(X).toarray()


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# Applying the machine learning algorithm from now


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,f1_score,classification_report

classifier=SVC()

classifier.fit(X_train,y_train.values.ravel())

y_pred=classifier.predict(X_test)

# performance analysis

print('Accuracy score : ',accuracy_score(y_test,y_pred))
print('ROC score : ',roc_auc_score(y_test,y_pred))
print('\nConfusion Matrix : \n',confusion_matrix(y_test,y_pred))
print('\n\n',classification_report(y_test,y_pred))
