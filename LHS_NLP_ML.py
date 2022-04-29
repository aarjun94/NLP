#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

df_training_tweet = pd.read_csv("/Users/arjunanandapadmanabhan/Downloads/wn22_data/wn22_PA_training_tweets.txt", sep=",")
df_labels = pd.read_csv("/Users/arjunanandapadmanabhan/Downloads/wn22_data/wn22_PA_training_labels.txt", sep=",")
final_df = pd.merge(df_training_tweet, df_labels, on="TweetID")
final_df




df_testing_tweet = pd.read_csv("/Users/arjunanandapadmanabhan/Downloads/wn22_data/wn22_PA_testing_tweets.txt", sep=",")

df_testing_tweet



import re
import nltk
from nltk.corpus import stopwords

def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda x: re.sub(r"(@[A-Za-z0-9]+)|(\d+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", x)) 
    return df
test_clean = clean_text(df_testing_tweet, "Tweet")
train_clean = clean_text(final_df, "Tweet")

train_clean.Tweet[4]




import pandas as pd
import numpy as np
import nltk
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

train_clean['no_contract'] = train_clean['Tweet'].apply(lambda x: [contractions.fix(word) for word in x.split()])
train_clean['Tweet'] = [' '.join(map(str, l)) for l in train_clean['no_contract']]

test_clean['no_contract'] = test_clean['Tweet'].apply(lambda x: [contractions.fix(word) for word in x.split()])
test_clean['Tweet'] = [' '.join(map(str, l)) for l in test_clean['no_contract']]





train_clean['tokenized'] = train_clean['Tweet'].apply(word_tokenize)
train_clean['tokenized'] = train_clean['tokenized'].apply(lambda x: [word.lower() for word in x])

test_clean['tokenized'] = test_clean['Tweet'].apply(word_tokenize)
test_clean['tokenized'] = test_clean['tokenized'].apply(lambda x: [word.lower() for word in x])





punc = string.punctuation
train_clean['no_punc'] = train_clean['tokenized'].apply(lambda x: [word for word in x if word not in punc])

test_clean['no_punc'] = test_clean['tokenized'].apply(lambda x: [word for word in x if word not in punc])




stop_words = set(stopwords.words('english'))
train_clean['tokenized']  = train_clean['no_punc'] .apply(lambda x: [word for word in x if word not in stop_words])

test_clean['tokenized']  = test_clean['no_punc'] .apply(lambda x: [word for word in x if word not in stop_words])





train_clean['pos_tags'] = train_clean['tokenized'].apply(nltk.tag.pos_tag)

test_clean['pos_tags'] = test_clean['tokenized'].apply(nltk.tag.pos_tag)





def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
train_clean['wordnet_pos'] = train_clean['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])

test_clean['wordnet_pos'] = test_clean['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])





wnl = WordNetLemmatizer()
train_clean['lemmatized'] = train_clean['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

test_clean['lemmatized'] = test_clean['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])

train_clean['Tweet'] = [' '.join(map(str, l)) for l in train_clean['lemmatized']]

test_clean['Tweet']  = [' '.join(map(str, l)) for l in test_clean['lemmatized']]


# Can be used to check the frequency of a term

# d = {}
# for word in train_clean['Tweet']:
#     for item in word.split():
#         if item in d:
#             d[item] = d[item] + 1
#         else:
#             d[item] = 1



# for i, word in enumerate(train_clean['Tweet']):
#     for item in word.split():
#         if d[item] <=10:
#             train_clean['Tweet'][i] = train_clean['Tweet'][i].replace(item, '')
#             train_clean['Tweet'][i] = re.sub(' +', ' ', train_clean['Tweet'][i])
            
            
# train_clean['Tweet'][0]




from sklearn.utils import resample
train_majority = train_clean[train_clean.Label==0]
train_minority = train_clean[train_clean.Label==1]
train_minority_upsampled = resample(train_minority, 
                                 replace=True,    
                                 n_samples=len(train_majority),   
                                 random_state=123)
train_upsampled = pd.concat([train_minority_upsampled, train_majority])
train_upsampled['Label'].value_counts()




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
# from sklearn import tree
# pipeline_svc = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf',  TfidfTransformer()),
#     ('nb', SGDClassifier(learning_rate = 'constant', eta0=0.96, epsilon=0.0004, max_iter=5000, validation_fraction=0.8, loss='log')),
# ])

# pipeline_svc = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf',  TfidfTransformer()),
#     ('nb', MLPClassifier(hidden_layer_sizes = (2,),solver='adam', learning_rate='adaptive', max_iter=1000, epsilon=1e-6, tol=0.0001))
# ])


# pipeline_svc = Pipeline([
#     ('vect', CountVectorizer(ngram_range=(1,3) )),
#     ('tfidf',  TfidfTransformer()),
#     ('nb', svm.SVC(kernel='poly', C=2.5, degree=2, coef0=0.18, break_ties=True))
# ])

pipeline_svc = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('nb', svm.SVC(gamma='scale'))
])

# pipeline_svc = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf',  TfidfTransformer()),
#     ('nb', GradientBoostingClassifier(n_estimators=10000, subsample=1))
# ])



# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# from numpy import arange

# ep = [0.001, 0.0002, 0.0003, 0.0004, 0.0007, 0.005, 0.03, 0.01, 0.003, 0.000004, 0.00000006, 0.005]
# max_num = 0.0
# ep_max = 0.0
# et_max = 0.0
# for e in ep:
#     for et in arange(0.01,1, 0.01):
#         pipeline_svc = Pipeline([
#         ('vect', CountVectorizer()),
#         ('tfidf',  TfidfTransformer()),
#         ('nb', SGDClassifier(learning_rate = 'constant', eta0=et, epsilon=e, max_iter=5000, validation_fraction=0.8, loss='log')),
#     ])

#         X_train, X_test, y_train, y_test = train_test_split(train_upsampled['Tweet'], train_upsampled['Label'],random_state = 0)
#         model = pipeline_svc.fit(X_train, y_train)
#         y_predict = model.predict(X_test)

#         x = f1_score(y_test, y_predict)
#         if x > max_num:
#             max_num = x
#             ep_max = e
#             et_max = et
            
            
# print(max_num, ep_max, et_max)


# In[124]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(train_upsampled['Tweet'], train_upsampled['Label'],random_state = 123)


# In[137]:


X_train = train_upsampled['Tweet']
y_train = train_upsampled['Label']


# In[138]:


model = pipeline_svc.fit(X_train, y_train)
# y_predict = model.predict(X_test)
# from sklearn.metrics import f1_score
# f1_score(y_test, y_predict)


# In[140]:


x_valid = df_testing_tweet['Tweet']
y_predict_1 = model.predict(x_valid)
y_predict_1


# In[141]:


# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vec = TfidfVectorizer()

# x_training = tfidf_vec.fit_transform(X_train)
# x_validation = tfidf_vec.transform(X_test)


# In[142]:


# from sklearn import svm

# model_svm = svm.SVC()

# model_svm.fit(x_training,y_train)


# In[143]:


# pred_svm = model_svm.predict(x_validation)


# In[144]:


# from sklearn.metrics import f1_score
# f1_score(y_test, pred_svm)


# In[145]:


# x_valid = df_testing_tweet['Tweet']
# x_test = tfidf_vec.transform(x_valid)
# y_predict_1 = model_svm.predict(x_test)
# y_predict_1


# In[146]:


pred = pd.DataFrame(y_predict_1)
pred.columns = ['Label']
pred



data = [df_testing_tweet['TweetID'], pred['Label']]
headers = ["TweetID", "Label"]

Final = pd.concat(data, axis=1, keys=headers)

Final.to_csv(r'/Users/arjunanandapadmanabhan/Downloads/wn22_data/output_2.txt', header=True, index=None, sep=',', mode='a')


# Confusion Matrix


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




classes=list(set(y_test))
cm = confusion_matrix(y_test, y_predic, labels=classes)
plot_confusion_matrix(cm, classes=classes, title='SGD')






