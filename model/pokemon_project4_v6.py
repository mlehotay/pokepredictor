#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import os
import catboost as cb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer,  StandardScaler, LabelEncoder

import sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper


import pickle
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.metrics import classification_report

pd.set_option('display.max_rows', 100)
#np.set.printoptions(prediction = 4)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#df = pd.read_csv('data/pokemon-go.csv')


df = pd.read_csv('data/pokemon_go.csv')
df_pokedex_name = pd.read_csv('data/pokedex.csv')


# In[ ]:


#df_pokedex_name.shape


# In[ ]:


#df_pokedex_name.head()


# In[ ]:


#df_pokedex_name.isnull().sum()


# In[ ]:


df1 = df_pokedex_name.drop('img', axis = 1)
df2 = df1.drop('wiki', axis = 1)
df3 = df2.drop('type_2', axis = 1)
df4 = df3.drop('name', axis = 1)


# In[ ]:


df3.info()


# In[ ]:


df = df.merge(df4)  ###merge 2 data frames


# In[ ]:


df


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


#split_1


# In[ ]:


X_split_train  = X_train.assign(first_part=split[0], last_part=split[1])
X_split_test  = X_test.assign(first_part=split_1[0], last_part=split_1[1])


# In[ ]:


#X_split_test


# In[ ]:


X_split_train.drop('local_time', 1, inplace=True)
X_split_test.drop('local_time', 1, inplace=True)


# In[ ]:


X_train = X_split_train
X_test = X_split_test


# In[ ]:


#X_test


# In[ ]:


X_train = X_train.rename(columns = {'last_part':'local_time'})
X_test = X_test.rename(columns = {'last_part':'local_time'})


# In[ ]:


X_train = X_train.rename(columns = {'first_part':'week_day'})
X_test = X_test.rename(columns = {'first_part':'week_day'})


# In[ ]:


X_test.head()


# In[ ]:


X_train['day_of_week']  = pd.to_datetime(X_train['week_day']).dt.weekday_name


# In[ ]:


#X_train['day_of_week'] = (X_train['week_day']).dt.weekday_name


# In[ ]:


X_test['day_of_week']  = pd.to_datetime(X_test['week_day']).dt.weekday_name


# In[ ]:


X_train.drop('latitude', 1, inplace=True)
X_test.drop('latitude', 1, inplace=True)

X_train.drop('longitude', 1, inplace=True)
X_test.drop('longitude', 1, inplace=True)


# In[ ]:


X_train.drop('week_day', 1, inplace=True)
X_test.drop('week_day', 1, inplace=True)


# In[ ]:


X_test


# In[ ]:


### save modified X and y  data frames
#export_csv = df.to_csv (r'data\modified_pokemon_go.csv', index = None, header=True) 


# In[ ]:


X_test['local_time']


# In[ ]:


dt = X_train['local_time']


# In[ ]:


dt_test = X_test['local_time']


# In[ ]:


dt_test


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


X_test


# In[ ]:


X_train = X_train.join(dtime, how = 'right')


# In[ ]:


X_test = X_test.join(dtime_test, how = 'right')


# In[ ]:


#X_train


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





# In[ ]:


"""
X_train['local_time'] = pd.cut(X_train.local_time, bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4])
X_test['local_time'] = pd.cut(X_test.local_time, bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4])
X_train['local_time'] = X_train['local_time'].apply({1:'Night', 2 :'Morning', 3 :'Afternoon', 4 :'Evening'}.get)
X_test['local_time'] = X_test['local_time'].apply({1 : 'Night', 2 : 'Morning', 3 : 'Afternoon', 4 :'Evening'}.get)
"""


# In[ ]:


#X_test


# In[ ]:


#X_test


# In[ ]:


#X_train.info()


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
    (['pokemon_type'], [LabelBinarizer()]),
    (['weather'], [LabelBinarizer()]),
    (['close_to_water'], [LabelBinarizer()]),
    (['temperature'], [StandardScaler()]),
    (['population_density'], [StandardScaler()]),
    (['day_of_week'], [LabelBinarizer()]),
    (['local_time'],  [SimpleImputer(), StandardScaler()]),
    ], df_out= True)


# In[ ]:


mapper


# In[ ]:


#X_train.shape


# In[ ]:


Z_train= mapper.fit(X_train)
#Z_test = mapper.transform(X_test)


# In[ ]:


#Z_train


# In[ ]:


Z_train= mapper.transform(X_train)


# In[ ]:


Z_test = mapper.transform(X_test)


# ### CAT BOOST CLASSIFIER

# In[ ]:


model = cb.CatBoostClassifier(
    iterations=10, 
    early_stopping_rounds=10,
    custom_loss=['AUC', 'Accuracy'])


# In[ ]:


model.fit(
    Z_train, 
    y_train,
    eval_set=(Z_test, y_test),
    verbose=False,
    plot=True)


# In[ ]:


train_score = model.score(Z_train, y_train) # train (learn) score


# In[ ]:


train_score


# In[ ]:


val_score = model.score(Z_test, y_test) # val (test) score


# In[ ]:


val_score

