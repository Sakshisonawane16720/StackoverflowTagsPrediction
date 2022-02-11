# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 22:36:07 2021

@author: Sakshi Sonawane
"""

# load data

import pandas as pd
df = pd.read_csv("dataset.csv")
print(df.columns)
df.drop('Record Count', axis=1)

# Pre-processing/Cleaing data

import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    
    text_stripwhitespaces = text.strip()        # Strip extra white spaces from the ends.
    text_stripwhitespaces = text_stripwhitespaces.replace("\\", " ")
    text_clean =  re.sub('[~`!@#$%^&*():;"{}_/?><\|.,`0-9]', '', text_stripwhitespaces.replace('-', ' ')) # Remove all the punctuation marks
    
    # Create tokens to perform lemmatiazation(turn all characters to lower case)
    tokens = word_tokenize(str(text_clean).lower())
    
    # Lemmatize each word and then join them(also remove stopwords)
    words = [lemmatizer.lemmatize(word) for word in tokens if not word in stop_words]
    if words[0] == words[-1]:
        words.pop(-1)
    final_text = ' '.join(words)
    return final_text

df['cleantext'] = df['post'].apply(lambda x: clean_text(x))


# Transformation

clsVars = "tags"
txtVars = "cleantext"

# Check for null values
print(df.isnull().sum()) 

# EDA

print(df.info())
print(df.head())

# Counts by Category
print(df.groupby('tags').size())

# dropping duplicates
u = df['post'].unique()
print(u.size)

#df = df.drop_duplicates()
#print(df.info())

# VDA

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 5)
import seaborn as sns

# class count plot
print("\n*** Distribution Plot ***")
plt.figure()
sns.countplot(df[clsVars],label="Count")
plt.title('Class Variable')
plt.show()

# split data into train and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['cleantext'],df['tags'], stratify=df['tags'],
                                test_size=0.33, random_state=707)
print("\n*** Length Of Train & Test Data ***")
print("X_train: ", len(X_train))
print("X_test : ", len(X_test))
print("y_train: ", len(y_train))
print("y_test : ", len(y_test))


# Analysis to find correlation between features and category

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(X_train).toarray()
labels = y_train
features.shape

# N = 5
# c_list = ['sport','business','tech','politics','entertainment']
# for category in c_list:
#   features_chi2 = chi2(features, labels == category)
#   indices = np.argsort(features_chi2[0])
#   feature_names = np.array(tfidf.get_feature_names())[indices]
#   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#   print("#'{}':".format(category))
#   print(" . Most correlated unigrams:\n\t. {}".format('\n\t. '.join(unigrams[-N:])))
#   print(" . Most correlated bigrams:\n\t. {}".format('\n\t. '.join(bigrams[-N:])))


# Comparing model scores

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    DecisionTreeClassifier(),
    KNeighborsClassifier()
]

# Cross Validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

print(cv_df.groupby('model_name').accuracy.mean())


# Prepare the selected model

model = LinearSVC()

model.fit(features, labels)
ft = tfidf.transform(X_test)
y_pred = model.predict(ft)


# Evaluting model

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

accuracy = accuracy_score(y_test, y_pred)*100
print("\n*** Accuracy ***")
print(accuracy)

# confusion matrix
# X-axis Actual | Y-axis Actual - to see how cm of original is
cm = confusion_matrix(y_test, y_test)
print("\n*** Confusion Matrix - Original ***")
print(cm)

# confusion matrix
# X-axis Predicted | Y-axis Actual
cm = confusion_matrix(y_test, y_pred)
print("\n*** Confusion Matrix - Predicted ***")
print(cm)

# classification report
print("\n*** Classification Report ***")
cr = classification_report(y_test,y_pred)
print(cr)


#Create model
model = LinearSVC()
model.fit(features, labels)

import pickle
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))

filename = 'vars.pkl'
pickle.dump(tfidf,open(filename,'wb'))
