#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import os
import catboost as cb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor

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

import pickle
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error


pd.set_option('display.max_rows', 100)
#np.set.printoptions(prediction = 4)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#df = pd.read_csv('data/pokemon-go.csv')


df = pd.read_csv('data/pokemon_go.csv')


# In[ ]:


df


# In[ ]:


#df =  df.sample(n=5000, random_state=42)


# In[ ]:


#df


# In[ ]:


df.isnull().sum()


# In[ ]:


df['weather'].value_counts()


# In[ ]:


df['close_to_water'].value_counts()


# In[ ]:


df['local_time'].value_counts()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[3]:


y = df[['pokedex_id']]


# In[4]:


X = df.drop('pokedex_id', axis = 1)


# In[ ]:


###df['previous_year'] = [row-1 for row in df['year']]


# In[ ]:


l = len(y)
l


# In[ ]:


y


# In[5]:


y_new = [[1 if v ==34  else 0 for v in y['pokedex_id']]]


# In[ ]:


y_new


# In[6]:


y_newdf = pd.DataFrame(y_new) ### convert list into dF


# In[7]:


y_newdf.isnull().sum()


# In[ ]:


#y_newdf.T


# In[8]:


y_pikachu = y_newdf.T


# In[10]:


y_pikachu


# In[11]:


y_pikachu[0].sum()


# In[ ]:





# In[ ]:


X= df.drop('pokedex_id', axis = 1)


# In[ ]:


X.info()


# In[ ]:


split = X['local_time'].str.split('T', 1, expand= True)


# In[ ]:


split


# In[ ]:


X_split = X.assign(first_part=split[0], last_part=split[1])


# In[ ]:


X_split.info()


# In[ ]:


X_split.drop('local_time', 1, inplace=True)


# In[ ]:


X_split.drop('first_part', 1, inplace=True)


# In[ ]:


X = X_split


# In[ ]:


split = X['local_time'].str.split(':', 1, expand= True)


# In[ ]:


X_split = X.assign(first_part=split[0], last_part=split[1])


# In[ ]:





# In[ ]:





# In[ ]:


y


# In[ ]:


### save modified X and y  data frames
export_csv = df.to_csv (r'data\modified_pokemon_go.csv', index = None, header=True) 


# In[ ]:


X.columns


# In[ ]:


X.info()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


# ### Making a model

# In[ ]:


df['weather'] = df['weather'].apply(lambda w: w.split("and"))
weather_types = {
'Breezy',
'Clear',
'DangerouslyWindy',
'Drizzle',
'Dry',
'Foggy',
'HeavyRain',
'Humid',
'LightRain',
'MostlyCloudy',
'Overcast',
'PartlyCloudy',
'Rain',
'RainandWindy',
'Windy'
}


# In[ ]:


#le = preprocessing.LabelEncoder()
#le.fit(X['local_time'])
#X['local_time'] = X['local_time'].astype('float')


# In[ ]:


mapper = DataFrameMapper([
    (['city'], [LabelBinarizer()]),
    (['weather'], [MultiLabelBinarizer(weather_types)]),
    (['close_to_water'], [LabelBinarizer()]),
    (['latitude'], [StandardScaler()]),
    (['longitude'], [StandardScaler()]),
    (['temperature'], [StandardScaler()]),
    (['population_density'], [StandardScaler()]),
    (['local_time'], [LabelBinarizer()]),
    ], df_out= True)


# In[ ]:


Z_train= mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)


# #### test new data for pipeline....1. make all the same modifications as for train df. Then proceed with below...

# ### CAT BOOST CLASSIFIER

# In[ ]:



df_cb.drop(df_cb.tail(1).index,inplace=True)   # drop last n rows


# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state=42)


# In[ ]:


cat_features = list(range(0, X.shape[1]))
print(cat_features)


# In[ ]:


df_cb.columns


# In[ ]:


print(f'Labels: {set(y)}')


# In[ ]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)


# In[ ]:


baseline_value = y_train.mean()


# In[ ]:


train_baseline = np.array([baseline_value] * y_train.shape[0])
test_baseline = np.array([baseline_value] * y_test.shape[0])


# In[ ]:


categorical_features_indices = np.where(X.dtypes != np.float)[0]
train_pool = Pool(X_train, y_train, baseline=train_baseline, cat_features=categorical_features_indices)


# In[ ]:


test_pool = Pool(X_test, y_test, baseline=test_baseline, cat_features=categorical_features_indices)


# In[ ]:


catboost_model = CatBoostRegressor(iterations=100, depth=2, loss_function="RMSE",verbose=False)


# In[ ]:


catboost_model.fit(train_pool, eval_set=test_pool)


# In[ ]:


preds1 = catboost_model.predict(test_pool)


# In[ ]:


preds2 = test_baseline + catboost_model.predict(X_test)


# In[ ]:


assert (np.abs(preds1 - preds2) < 1e-6).all()


# In[ ]:


print(mean_squared_error(y_test, preds1))

