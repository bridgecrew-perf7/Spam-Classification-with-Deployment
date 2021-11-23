
# Spam Classification

>Project status : Active and Incomplete


## Table of content :
* **Project intro**

        1. General info

        2. Methods used

        3. Technologies used

        4. Setup

        5. Dataset



* **Project Description**
* **Codes and technical aspects**
* **Deployment**

# 1. Project introduction

## General info

This project creates the machine learning model that helps us to predict whether the given text message is spam or not.

## Methods used

   * Machine Learning

   * Natural language processing

   * Predictive modeling

   * etc.


## Technologies used

This project is created using [**python**](https://www.python.org/) and other libraries like :

* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Sklearn/Scikit-Learn](https://scikit-learn.org/stable/#)
* [NLTK](https://www.nltk.org/)
* etc.

The other librarires are enlisted in the requirement.txt file.

## Setup

 
To run this project , install [**python**](https://www.python.org/) locally and install the requirements using [pip](https://pypi.org/project/pip/) or [conda](https://docs.conda.io/en/latest/) -

```terminal
	pip install -r requirement.txt
```

else you can use the [google colab notebook](Spam_Classifier.ipynb) for running the code online without installing any libraries , packages and dependencies.

## Dataset


The dataset used in this project is available in [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.



# 2.Project Description



# 3.Codes and Technical aspects

This project is entirely based on the machine learning and Natural language processing.

For that , we will be using **Python** Language as it is suitable and contains various libraries and packages  , and has easy syntax.
Alternatively , **R** language can also be used.


For running the project , python must be installed in your local system.
The required packages and libraries are listed in [requirement.txt]().

You can also use **Google-Colab** . **Colaboratory**, or “Colab” for short, is a product from Google Research. Colab allows anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education.
The colab notebook is also uploaded here.


The code simply contains multiple parts :

First of all ,we have to import all the required libraries like nltk, sklearn , etc.

And then , we have to import our dataset which contains the messages with the label of spam or not spam.

We have our dataset in the **csv** format . The v2 column contains message and v1 contains label.
We import our data using **pandas** library and save as the dataframe. 


The message we have in our dataset might not be clean . So we have to remove some unwanted stuffs like stopwords (i.e. "just" ,"is","oh","an" ,etc. ) and we also have to reduce the word into the word root form ( i.e. playing,plays,played to play, ).
we can do it so by using **Lamatization process**. The **NLTK** library provides the tool for that.


After this , we have to convert the words into vectors because the machine learning algorithm can't be directly fed with strings or characters . To we have to convert the sentences into vectors (i.e. numeric matrix )
This process is called feature extraction.
There are multiple tools for that like **Countvectorizer ,TF-IDF, word2vec**,etc. 
We are here using **Countvectorizer** which is one the widely used one.


Then , we split the dataset into training and testing (in the ratio of 7:3).

In the machine learning part , we are using **Multinomial Naive Baye's** algorithm to create a model.
This model preforms better in these cases and is also economical (in the case of time , memory and computational cost) than others.

We create an object of Multinomial Naive Baye`s algorithm and train it using the training dataset( both messages and labels ).


we predict the value for the testing messages (at this time only messages are passed not their labels) and compare with the original value/labels .
By this the proformance of the model is analyzed using various metrices like accuracy , confusion matrix , classification report , etc.


If the performance of the model is good . The model is ready to use and can be saved. 
Else the model needs to be re-trained ( by using another algorithm or by parameter tuning .)


Both model and the vectorizer needs to be saved for the deployment or future use.
