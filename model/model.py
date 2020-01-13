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

# get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data/pokemon_go.csv')

df

df.isnull().sum()

df['weather'].value_counts()

df['close_to_water'].value_counts()

df['local_time'].value_counts()

df.shape

df.isnull().sum()


y = df[['pokedex_id']]

X = df.drop('pokedex_id', axis = 1)

l = len(y)
l

y

y_new = [[1 if v ==34  else 0 for v in y['pokedex_id']]]

y_new

y_newdf = pd.DataFrame(y_new) ### convert list into dF

y_newdf.isnull().sum()

y_pikachu = y_newdf.T

y_pikachu

y_pikachu[0].sum()

X= df.drop('pokedex_id', axis = 1)

X.info()

# Remove local time
# split = X['local_time'].str.split('T', 1, expand= True)


# split
#
# X_split = X.assign(first_part=split[0], last_part=split[1])
#
# X_split.info()
#
# X_split.drop('local_time', 1, inplace=True)
#
# X_split.drop('first_part', 1, inplace=True)
#
# X = X_split
#
# split = X['local_time'].str.split(':', 1, expand= True)
#
# X_split = X.assign(first_part=split[0], last_part=split[1])

y

### save modified X and y  data frames
export_csv = df.to_csv (r'data\modified_pokemon_go.csv', index = None, header=True)

X.columns

X.info()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

# ### Making a model

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

weather_types = list(weather_types)

#le = preprocessing.LabelEncoder()
#le.fit(X['local_time'])
#X['local_time'] = X['local_time'].astype('float')

mapper = DataFrameMapper([
    (['city'], [LabelBinarizer()]),
    # (['weather'], [MultiLabelBinarizer(weather_types)]),
    (['close_to_water'], [LabelBinarizer()]),
    (['latitude'], [StandardScaler()]),
    (['longitude'], [StandardScaler()]),
    (['temperature'], [StandardScaler()]),
    (['population_density'], [StandardScaler()])
    # (['local_time'], [LabelBinarizer()]),
    ], df_out= True)

Z_train= mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)


# #### test new data for pipeline....1. make all the same modifications as for train df. Then proceed with below...

# ### CAT BOOST CLASSIFIER

# df_cb.drop(df_cb.tail(1).index,inplace=True)   # drop last n rows

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state=42)

# df_cb.columns

cat_features = list(range(0, X.shape[1]))
print(cat_features)

print(f'Labels: {set(y)}')

categorical_features_indices = np.where(X.dtypes != np.float)[0]
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)

baseline_value = y_train.mean()

train_baseline = np.array([baseline_value] * y_train.shape[0])
test_baseline = np.array([baseline_value] * y_test.shape[0])

categorical_features_indices = np.where(X.dtypes != np.float)[0]

train_pool = Pool(X_train, y_train, baseline=train_baseline, cat_features=categorical_features_indices)

test_pool = Pool(X_test, y_test, baseline=test_baseline, cat_features=categorical_features_indices)

catboost_model = CatBoostRegressor(iterations=20, depth=2, loss_function="RMSE",verbose=False)

catboost_model.fit(train_pool, eval_set=test_pool)

preds1 = catboost_model.predict(test_pool)

preds2 = test_baseline + catboost_model.predict(X_test)

assert (np.abs(preds1 - preds2) < 1e-6).all()

print(mean_squared_error(y_test, preds1))


# Pipe
pipe = make_pipeline(mapper, catboost_model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

#Pickle
pickle.dump(pipe, open('model/pipe.pkl', 'wb'))
del pipe
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
pipe
