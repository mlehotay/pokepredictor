#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
#import seaborn as sns
import numpy as np
import os
import catboost as cb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing

from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer,  StandardScaler, LabelEncoder

import sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper


import pickle
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.metrics import classification_report

pd.set_option('display.max_rows', 100)
#np.set.printoptions(prediction = 4)


# In[ ]:


df = pd.read_csv('data/pokemon_go.csv')


# In[ ]:


df['local_time'].value_counts()


# In[ ]:


df.shape


# In[ ]:


target = 'pokedex_id'
y = df[target]


# In[ ]:


len(y)


# In[ ]:


X = df.drop(target, axis = 1)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[ ]:


split = X_train['local_time'].str.split('T', 1, expand= True)
split_1 = X_test['local_time'].str.split('T', 1, expand= True)


# In[ ]:


X_split_train  = X_train.assign(first_part=split[0], last_part=split[1])
X_split_test  = X_test.assign(first_part=split_1[0], last_part=split_1[1])


# In[ ]:


X_split_train.drop('local_time', 1, inplace=True)
X_split_test.drop('local_time', 1, inplace=True)


# In[ ]:


X_train = X_split_train
X_test = X_split_test


# In[ ]:


X_train = X_train.rename(columns = {'last_part':'local_time'})
X_test = X_test.rename(columns = {'last_part':'local_time'})


# In[ ]:


X_train = X_train.rename(columns = {'first_part':'week_day'})
X_test = X_test.rename(columns = {'first_part':'week_day'})


# In[ ]:


X_train['day_of_week']  = pd.to_datetime(X_train['week_day']).dt.weekday_name


# In[ ]:


X_test['day_of_week']  = pd.to_datetime(X_test['week_day']).dt.weekday_name



# In[ ]:


X_train.head()


# In[ ]:


X_train.drop('latitude', 1, inplace=True)
X_test.drop('latitude', 1, inplace=True)

X_train.drop('longitude', 1, inplace=True)
X_test.drop('longitude', 1, inplace=True)

X_train.drop('population_density', 1, inplace=True)
X_test.drop('population_density', 1, inplace=True)


# In[ ]:


X_train.drop('week_day', 1, inplace=True)
X_test.drop('week_day', 1, inplace=True)


# In[ ]:


#X_test


# In[ ]:


### save modified X and y  data frames
#export_csv = df.to_csv (r'data\modified_pokemon_go.csv', index = None, header=True)


# In[ ]:


dt = X_train['local_time']


# In[ ]:


dt_test = X_test['local_time']


# In[ ]:


class DateFormatter(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xtime = X.apply(pd.to_datetime)
        return Xtime


class DateEncoder(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt = X.dt
        return pd.concat([dt.hour, dt.minute, dt.second], axis=1)


# In[ ]:


DATE_COLS = ['local_time']

datemult = DataFrameMapper(
            [ (i,[DateFormatter(),DateEncoder()]) for i in DATE_COLS     ]
            , input_df=True, df_out=True)

dtime = datemult.fit_transform(X_train)


# In[ ]:


DATE_COLS = ['local_time']

datemult = DataFrameMapper(
            [ (i,[DateFormatter(),DateEncoder()]) for i in DATE_COLS     ]
            , input_df=True, df_out=True)

dtime_test = datemult.fit_transform(X_test)


# In[ ]:


dtime_test


# In[ ]:


X_train = X_train.join(dtime, how = 'right')


# In[ ]:


X_test = X_test.join(dtime_test, how = 'right')


# In[ ]:


X_train


# In[ ]:


X_train.drop('local_time', 1, inplace=True)
X_test.drop('local_time', 1, inplace=True)


# In[ ]:


X_train.drop('local_time_1', 1, inplace=True)
X_test.drop('local_time_1', 1, inplace=True)


# In[ ]:


X_train.drop('local_time_2', 1, inplace=True)
X_test.drop('local_time_2', 1, inplace=True)


# In[ ]:


X_train = X_train.rename(columns = {'local_time_0':'local_time'})
X_test = X_test.rename(columns = {'local_time_0':'local_time'})

X_train = X_train.rename(columns = {'type_1':'pokemon_type'})
X_test = X_test.rename(columns = {'type_1':'pokemon_type'})


# In[ ]:


X_test.info()


# ### Making a model

# In[ ]:


le = preprocessing.LabelEncoder()
#le.fit(X_train['local_time'])
le.fit(X_train['day_of_week'])
#le.fit(X_test['local_time'])
le.fit(X_test['day_of_week'])


# In[ ]:


mapper = DataFrameMapper([
    (['city'], [LabelBinarizer()]),
    (['weather'], [LabelBinarizer()]),
    (['close_to_water'], [LabelBinarizer()]),
    (['temperature'], [StandardScaler()]),
    (['day_of_week'], [LabelBinarizer()]),
    (['local_time'],  [SimpleImputer(), StandardScaler()]),
    ], df_out= True)


# In[ ]:


mapper


# In[ ]:


Z_train= mapper.fit(X_train)


# In[ ]:


Z_train= mapper.transform(X_train)


# In[ ]:


Z_test = mapper.transform(X_test)


# ### Logistic Regression

# In[ ]:


"""
CAT BOOST CLASSIFIER

model = cb.CatBoostClassifier(
    iterations=5,
    early_stopping_rounds=10,
    custom_loss=['AUC', 'Accuracy'])

    model.fit(
    Z_train,
    y_train,
    eval_set=(Z_test, y_test),
    verbose=False,
    plot= False)

    train_score = model.score(Z_train, y_train) # train (learn) score
    val_score = model.score(Z_test, y_test) # val (test) score
    print(train_score, val_score)
 """


# In[ ]:


model = LogisticRegression(solver='lbfgs', max_iter = 100)
#model.fit(Z_train, y_train)


# In[ ]:


#y_pred = model.predict(Z_test)


# In[ ]:





# In[ ]:


#from sklearn.metrics import confusion_matrix
#confusion_matrix = confusion_matrix(y_test, y_pred)
#confusion_matrix


# In[ ]:


#print(classification_report(y_test, y_pred))


# In[ ]:


# Pipe
pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# In[ ]:


#Pickle
pickle.dump(pipe, open('model/pipe.pkl', 'wb'))
del pipe
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
pipe
