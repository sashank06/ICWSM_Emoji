#!/usr/bin/env python
# coding: utf-8

# In[1]:
__author__ = ["Sashank Santhanam"]
__credits__ = ["Sashank Santhanam"]
__maintainer__ = "Sashank Santhanam"
__email__ = "ssantha1@uncc.edu"

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import TweetTokenizer
from numpy import loadtxt
from pathlib import Path
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import pandas as pd
from sklearn.svm import SVC
import nltk
from textblob import TextBlob #used this package to do textmining and calculate sentiment
from nltk.corpus import stopwords #to pre-process the data
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import time
import json
import glob
from collections import Counter
from sklearn.metrics import confusion_matrix,accuracy_score
import itertools
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#same code applicable for paris and irma
df = pd.read_csv('/Users/RSANTHANAM/Desktop/Solidarity/PARIS/paris_analysis.csv')
tweets = df['tweet'].tolist()
print("Number of tweets: {}".format(len(tweets)))

def remove_http(tweets):
    removed = []
    for tweet in tweets:
        text = re.sub(r"http\S+", "", str(tweet))
        #text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"",str(tweet))
        #text = text.encode('ISO-8859-1').decode('utf-8','ignore')
        #text = text.encode('utf-8').decode('utf-8','ignore')
        text = text.encode('utf-8').decode('utf-8')
        removed.append(text)
    return removed




tweets_unique = remove_http(tweets)
df_annotations = pd.read_excel('/Users/RSANTHANAM/Desktop/Solidarity/PARIS/annotations_new.xlsx')
df_annotations= df_annotations[['hashtags','annotations']]

df_solidarity = df_annotations.loc[df_annotations['annotations'] == "solidarity"]
df_nosolidarity = df_annotations.loc[df_annotations['annotations'] == "no solidarity"]

solidarity = df_solidarity['hashtags'].tolist()
no_solidarity = df_nosolidarity['hashtags'].tolist()


tweets = list(set(tweets_unique))

print("Number of unique tweets: {}".format(len(tweets)))


# In[22]:


def get_hashtags(tweets):
    hashtag = []
    for tweet in tweets:
        #hash_ = []
        hashtag.append(re.findall(r"#(\w+)", tweet.lower()))
        #hashtag.append(hash_
    return hashtag
hashtags_list = get_hashtags(tweets)


def annotate_tweet(hashtags_list):
    annotated_list = []
    for hashtag in hashtags_list:
        switch = 2
        for tag in hashtag:
            #if tag not in solidarity and tag not in no_solidarity:
            #    switch = 0
            #else:
            if tag in solidarity and tag not in no_solidarity:
                switch = 1
            elif tag in no_solidarity and tag not in solidarity:
                switch = -1
            elif tag not in no_solidarity and tag not in solidarity and switch==1:
                switch = 1
            elif tag in no_solidarity and tag not in solidarity and switch==-1:
                switch = -1
            elif tag in no_solidarity and tag not in solidarity and switch==1:
                switch = 0
                break
            elif tag not in no_solidarity and tag in solidarity and switch==-1:
                switch = 0
                break
            else:
                switch = 0
        if switch == -1:
            annotated_list.append(-1)
        elif switch == 1:
            annotated_list.append(1)
        else:
            annotated_list.append(0)
    return annotated_list


# In[27]:


annotated_list = annotate_tweet(hashtags_list)



df['annotations'] = pd.DataFrame(annotated_list)


# # get the solidarity and non solidarity tweets

# In[33]:


def tweets_labels(tweets,annotated_list):
    tweet = []
    labels = []
    for i in range(0,len(tweets)):
        if annotated_list[i] == 1 or annotated_list[i] == -1:
            labels.append(annotated_list[i])
            tweet.append(tweets[i])
        else:
            pass
    return labels,tweet


# In[35]:


labels,tweet = tweets_labels(tweets,annotated_list)

print("Number of tweets: {}".format(len(tweet))) #50339 - paris
print("Number of Solidarity englishtweets: {}".format(labels.count(1))) #20465
print("Number of Non-Solidarity englishtweets: {}".format(labels.count(-1))) #29874


# In[41]:


df_english = pd.DataFrame(index=range(len(tweet)))

df_english['tweets'] = pd.DataFrame(tweet)
df_english['labels'] = pd.DataFrame(labels)
df_english.to_csv('latest_paris_unique.csv',index=False)
rus= RandomUnderSampler()


# In[78]:


tweet_2D = []
for tw in tweet:
    tweet_2D.append([tw])

tweet_sampled,labels_sampled = rus.fit_sample(tweet_2D,labels)
plt.hist(labels_sampled)
plt.show()

tweet_sampled_1D = []
for tweet in tweet_sampled:
    for tw in tweet:
        tweet_sampled_1D.append(tw)

tweet_sampled_1D_nohttp = remove_http(tweet_sampled_1D)

def remove_hashtags(tweets):
    hashremoved_tweets = []
    for tweet in tweets:
        hashtags_list_final = []
        #hashtags_list_final.append(re.findall(r"#(\w+)", tweet.lower()))
        hashtags_list_final = re.findall(r"#(\w+)", tweet.lower())
        #resultwords  = [word for word in tweet if word.lower() not in stopwords]
        #result = ' '.join(resultwords)
        resultwords = []
        for word in tweet.split():
            if word.lower().startswith("#"):
                #if word.lower() in hashtags_list_final:
                string = word.replace("#","")
                if (string.lower() in solidarity or string.lower() in no_solidarity):
                    pass
                else:
                    string = "#"+ string.lower()
                    resultwords.append(string.lower())
            else:
                resultwords.append(word.lower())
        result = ' '.join(resultwords)        
        #print(hashtags_list_final)
        hashremoved_tweets.append(result)
    return hashremoved_tweets
    #break


# In[85]:


final_english_tweets = remove_hashtags(tweet_sampled_1D_nohttp)



X_train, X_test, Y_train, Y_test = train_test_split(final_english_tweets, labels_sampled, test_size=0.2, random_state=42)



count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)





tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)





#change this to experiment out with features
text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english',ngram_range=(1,1))),

                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),])

_ = text_clf_svm.fit(X_train, Y_train)

predicted_svm = text_clf_svm.predict(X_test)


np.mean(predicted_svm == Y_test)
from sklearn.metrics import precision_recall_fscore_support
output = precision_recall_fscore_support(Y_test, predicted_svm)
print("Precision: {}".format(output[0]))
print("Recall: {}".format(output[1]))
print("F-Score: {}".format(output[2]))
print("Support: {}".format(output[3]))


# In[145]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[146]:


plt.figure(figsize=(8,8))
plot_confusion_matrix(confusion_matrix(Y_test, predicted_svm), classes=['solidarity','no solidarity'],
                      title='linear svm Confusion matrix')

plt.show()

