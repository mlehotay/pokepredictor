#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# import seaborn as sns
import numpy as np
import os
import catboost as cb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
# import matplotlib.pyplot as plt
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
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper


import pickle
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.metrics import classification_report

pd.set_option('display.max_rows', 100)
#np.set.printoptions(prediction = 4)

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('data/pokemon_go.csv')


# In[3]:


df['local_time'].value_counts()


# In[4]:


df.shape


# In[5]:


target = 'pokedex_id'
y = df[target]


# In[6]:


len(y)


# In[7]:


X = df.drop(target, axis = 1)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# In[9]:


split = X_train['local_time'].str.split('T', 1, expand= True)
split_1 = X_test['local_time'].str.split('T', 1, expand= True)


# In[10]:


X_split_train  = X_train.assign(first_part=split[0], last_part=split[1])
X_split_test  = X_test.assign(first_part=split_1[0], last_part=split_1[1])


# In[11]:


X_split_train.drop('local_time', 1, inplace=True)
X_split_test.drop('local_time', 1, inplace=True)


# In[12]:


X_train = X_split_train
X_test = X_split_test


# In[13]:


#X_test


# In[14]:


X_train = X_train.rename(columns = {'last_part':'local_time'})
X_test = X_test.rename(columns = {'last_part':'local_time'})


# In[15]:


X_train = X_train.rename(columns = {'first_part':'week_day'})
X_test = X_test.rename(columns = {'first_part':'week_day'})


# In[16]:


#X_test.head()


# In[17]:


X_train['day_of_week']  = pd.to_datetime(X_train['week_day']).dt.weekday_name


# In[18]:


X_test['day_of_week']  = pd.to_datetime(X_test['week_day']).dt.weekday_name



# In[19]:


X_train.head()


# In[20]:


X_train.drop('latitude', 1, inplace=True)
X_test.drop('latitude', 1, inplace=True)

X_train.drop('longitude', 1, inplace=True)
X_test.drop('longitude', 1, inplace=True)

X_train.drop('population_density', 1, inplace=True)
X_test.drop('population_density', 1, inplace=True)


# In[21]:


X_train.drop('week_day', 1, inplace=True)
X_test.drop('week_day', 1, inplace=True)


# In[22]:


#X_test


# In[23]:


### save modified X and y  data frames
#export_csv = df.to_csv (r'data\modified_pokemon_go.csv', index = None, header=True)


# In[24]:


dt = X_train['local_time']


# In[25]:


dt_test = X_test['local_time']


# In[26]:


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


# In[27]:


DATE_COLS = ['local_time']

datemult = DataFrameMapper(
            [ (i,[DateFormatter(),DateEncoder()]) for i in DATE_COLS     ]
            , input_df=True, df_out=True)

dtime = datemult.fit_transform(X_train)


# In[28]:


DATE_COLS = ['local_time']

datemult = DataFrameMapper(
            [ (i,[DateFormatter(),DateEncoder()]) for i in DATE_COLS     ]
            , input_df=True, df_out=True)

dtime_test = datemult.fit_transform(X_test)


# In[29]:


dtime_test


# In[30]:


X_train = X_train.join(dtime, how = 'right')


# In[31]:


X_test = X_test.join(dtime_test, how = 'right')


# In[32]:


X_train


# In[33]:


X_train.drop('local_time', 1, inplace=True)
X_test.drop('local_time', 1, inplace=True)


# In[34]:


X_train.drop('local_time_1', 1, inplace=True)
X_test.drop('local_time_1', 1, inplace=True)


# In[35]:


X_train.drop('local_time_2', 1, inplace=True)
X_test.drop('local_time_2', 1, inplace=True)


# In[36]:


X_train = X_train.rename(columns = {'local_time_0':'local_time'})
X_test = X_test.rename(columns = {'local_time_0':'local_time'})

X_train = X_train.rename(columns = {'type_1':'pokemon_type'})
X_test = X_test.rename(columns = {'type_1':'pokemon_type'})


# In[37]:


X_test.info()


# ### Making a model

# In[38]:


le = preprocessing.LabelEncoder()
#le.fit(X_train['local_time'])
le.fit(X_train['day_of_week'])
#le.fit(X_test['local_time'])
le.fit(X_test['day_of_week'])


# In[39]:


mapper = DataFrameMapper([
    (['city'], [LabelBinarizer()]),
    (['weather'], [LabelBinarizer()]),
    (['close_to_water'], [LabelBinarizer()]),
    (['temperature'], [StandardScaler()]),
    (['day_of_week'], [LabelBinarizer()]),
    (['local_time'],  [SimpleImputer(), StandardScaler()]),
    ], df_out= True)


# In[40]:


mapper


# In[41]:


Z_train= mapper.fit(X_train)


# In[42]:


Z_train= mapper.transform(X_train)


# In[43]:


Z_test = mapper.transform(X_test)


# ### CAT BOOST CLASSIFIER

# In[44]:


model = cb.CatBoostClassifier(
    iterations=5,
    early_stopping_rounds=10,
    custom_loss=['AUC', 'Accuracy'])


# In[45]:


model.fit(
    Z_train,
    y_train,
    eval_set=(Z_test, y_test),
    verbose=False,
    plot=False)


# In[46]:


train_score = model.score(Z_train, y_train) # train (learn) score


# In[47]:


train_score


# In[48]:


val_score = model.score(Z_test, y_test) # val (test) score


# In[49]:


#val_score


# In[51]:


# Pipe
pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# In[53]:


#Pickle
pickle.dump(pipe, open('model/pipe.pkl', 'wb'))
del pipe
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
pipe


# In[ ]:





# In[ ]:





# In[ ]:
