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

dataset=pd.read_csv(r'spam.csv',encoding='latin-1')

X=dataset['v2']


# converting the label of the messages (i.e. ham or spam ) into zeros and ones

y=dataset['v1']
y=pd.get_dummies(y,drop_first=True)


# stop words of english languages
stopwords=stopwords.words('english')


# creating the object of the lamatizer
lemmatizer=WordNetLemmatizer()

n,=X.shape

## Data cleaning
for i in range(n):
    # substitution of all characters except alphabets
    temp=re.sub('[^a-zA-Z]',' ',X[i])
    
    #lowering all the words
    temp=temp.lower()
    
    #splitting each words
    temp=temp.split()
    
    # lamatize each word and removing the stopwords
    temp=[lemmatizer.lemmatize(word) for word in temp if word not in set(stopwords)]
    
    # joining the words into sentences
    temp=' '.join(temp)
    X[i]=temp



# Features extraction :: converting words into vectors
cv=CountVectorizer()

X=cv.fit_transform(X).toarray()


# splitting the dataset into training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# Applying the machine learning algorithm from now


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,f1_score,classification_report


# Using the MultiNomial naive bayes algorithm
classifier=MultinomialNB()

classifier.fit(X_train,y_train.values.ravel())

y_pred=classifier.predict(X_test)

# performance analysis

print('Accuracy score : ',accuracy_score(y_test,y_pred))
print('ROC score : ',roc_auc_score(y_test,y_pred))
print('\nConfusion Matrix : \n',confusion_matrix(y_test,y_pred))
print('\n\n',classification_report(y_test,y_pred))

# saving both model and vectorizer
import pickle
pickle.dump(model,'my_model.pkl')
pickle.dump(cv,'countvector.pkl')
