#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

os.chdir(r"C:\Users\randy\OneDrive\Desktop\Udacity\Identify Fraud from Enron Email")
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Function/File imports
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[2]:


### Classifiers in use
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import AdaBoostClassifier


# ## Task 1:  Select Features

# In[3]:


### features_list is a list of strings, each of which is a feature name.

features_list = ['poi', 'salary','bonus','other','total_stock_value','expenses','shared_receipt_with_poi'] 


# In[4]:


### Load the dictionary containing the dataset

data_dict = pd.read_pickle("final_project_dataset.pkl") ## Read data file with pandas
df = pd.DataFrame(data_dict).transpose() ## Convert to Dataframe and transpose for easier exploration
print (df.info())


# In[5]:


df = df.drop(['email_address'],axis=1, errors ='ignore') ## Email address will not be used
df = df.astype(float) ## Convert remaining data types
df.fillna(0, inplace = True)


# In[6]:


## Isolating the features by poi
total_poi = len(df.loc[df['poi'] == 1, 'poi' ])
total_non_poi = len(df.loc[df['poi'] == 0, 'poi' ])
percent_poi = (total_poi/(total_poi+total_non_poi))*100

poi_s_p = (df.loc[df['poi'] == 1, 'salary'].sum()/df.loc[df['poi'] == 0, 'salary'].sum())*100
poi_b_p = (df.loc[df['poi'] == 1, 'bonus'].sum()/df.loc[df['poi'] == 0, 'bonus'].sum())*100
poi_o_p = (df.loc[df['poi'] == 1, 'other'].sum()/df.loc[df['poi'] == 0, 'other'].sum())*100
poi_st_p = (df.loc[df['poi'] == 1, 'total_stock_value'].sum()/df.loc[df['poi'] == 0, 'total_stock_value'].sum())*100
poi_e_p = (df.loc[df['poi'] == 1, 'expenses'].sum()/df.loc[df['poi'] == 0, 'expenses'].sum())*100


print("Total percent of population that is poi =", percent_poi)
print("")
print ("poi percent of salary = ", poi_s_p)
print ("")
print ("poi percent of bonus = ", poi_b_p)
print ("")
print ("poi percent of other = ", poi_o_p)
print ("")
print ("poi percent of total stock value = ", poi_st_p)
print ("")
print ("poi percent of expenses = ", poi_e_p)



# ## Task 2: Remove the outliers from within the salary data

# In[7]:


### Plot the salary data
fig, ax = plt.subplots(figsize=(8,3))
ax.scatter(df['salary'],df['salary'])
x = df.salary
plt.show()


# In[8]:


## See the highest salary to find the outlier shown above
(df.sort_values(by = ['salary'],ascending=False).head(2))


# In[9]:


## Drop the TOTAL column as it is unrelated to the rest of the salary data
df.drop('TOTAL', axis=0, inplace = True, errors = 'ignore')

## Preview the data once again
fig, ax = plt.subplots(figsize=(8,3))
ax.scatter(df['salary'],df['salary'])
x = df.salary
plt.show()


# In[10]:


## Look at top values to look into more outliers

(df.sort_values(by = ['salary'],ascending=False).head(2))


# In[11]:


## Look into lower values, they are zero values but they do show significant stock value, so they are retained

(df.sort_values(by = ['salary'],ascending=True).head(5))


# ## Task 3: Create New Feature(s)

# In[12]:


### Emails are calculated as a percent of whole

df['percent_to_poi'] = (df['from_this_person_to_poi']/df['to_messages']) * 100
df['percent_from_poi'] = (df['from_poi_to_this_person']/df['from_messages']) * 100
df['percent_shared_from_poi'] = (df['shared_receipt_with_poi']/(df['from_messages'] + df['to_messages'])) * 100
df['total_stock_value'] = abs(df['total_stock_value'])
df.fillna(0, inplace = True)

## Add new features to a list and extend them onto existing

created_features = ['percent_to_poi', 'percent_shared_from_poi'] # new features to be added within existing
features_list.remove('shared_receipt_with_poi') # remove to replace with percent of whole value
features_list.extend(created_features)


# In[13]:


### Store to my_dataset for easy export below.

my_dataset = df.transpose()


# In[14]:


### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[15]:


## Split the data into train and test

cv = StratifiedShuffleSplit(n_splits=800,random_state = 42)
for train_idx, test_idx in cv.split(features,labels):
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )


# ## Task 4: Try A Variety Of Classifiers

# In[16]:


def algo_choice_split(algo):
    '''
    Creates classifier object, then prints out metric reports
    '''
    clf = algo
    clf.fit(features_train,labels_train)
    predictions = clf.predict(features_test)
    print("Algorithm: ", clf)
    print("Precision", round(metrics.precision_score(labels_test,predictions),2))
    print("Recall", round(metrics.recall_score(labels_test,predictions),2))
    print("Accuracy", round(metrics.accuracy_score(labels_test,predictions),2))
    print("")
    print(metrics.confusion_matrix(labels_test,predictions))
    print(metrics.classification_report(labels_test,predictions))
    print("")
    


# In[17]:


## Call functions for classifier fitting and scores

algo_choice_split(ComplementNB())
algo_choice_split(LogisticRegression())
algo_choice_split(AdaBoostClassifier())
algo_choice_split(RandomForestClassifier())


# ## Task 5: Tune classifier 

# In[18]:


def tuner (algo,param_grid):
    '''
    Performs GridSearchCV for testing multiple classifiers w/out redundancy
    '''
    clf = GridSearchCV(estimator = algo,
                       param_grid=param_grid,
                       cv=5,
                       n_jobs=-1)
    clf.fit(features_train,labels_train)
    predictions = clf.predict(features_test)
    print(clf.best_estimator_)


# In[19]:


## ComplementNB

NB_param_grid = {'alpha':  [.001, 0.01, 0.1, 0.5, 1.0, 10.0]} 

## call tuner function to find optimal parameter settings

tuner(ComplementNB(),NB_param_grid)
#Accuracy: 0.50913	Precision: 0.08147	Recall: 0.26100	F1: 0.12418	F2: 0.18116


# In[20]:


## AdaBoost

ada_param_grid = {
    'n_estimators': [18,20,22,25,27,30],
    'learning_rate': [.25,.3,.35,.4, .45, .5,],
    'random_state': [42]
}

## call tuner function to find optimal parameter settings

tuner(AdaBoostClassifier(),ada_param_grid)
#Accuracy: 0.86680	Precision: 0.50075	Recall: 0.33400	F1: 0.40072	F2: 0.35783


# In[21]:


## RandomForest

rf_param_grid = {
    'n_estimators': [50,100,200,300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': [42]
}

## call tuner function to find optimal parameter settings

tuner(RandomForestClassifier(),rf_param_grid)
#Accuracy: 0.86580	Precision: 0.49081	Recall: 0.17350	F1: 0.25637	F2: 0.19926


# In[22]:


## Final Classifier

clf=AdaBoostClassifier(learning_rate=0.3, n_estimators=25, random_state=42)


# ## Task 6: Dump classifier, dataset, and features_list

# In[23]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can

pickle.dump((clf),open("my_classifier.pkl","wb"))
pickle.dump((my_dataset),open("my_dataset.pkl","wb"))
pickle.dump((features_list),open("my_feature_list.pkl","wb"))


# In[ ]:




