import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn_pandas import DataFrameMapper, CategoricalImputer, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import r2_score
from sklearn import metrics

import pickle
import flask



df = pd.read_csv('data/pokemon_go.csv')


# 578 Pikachu
df.loc[(df['pokedex_id'] ==34)].sum()




df.tail(100)

df.tail()

df.info()

df.describe().T

df.groupby('city').count()


# Set of cities
cities = df['city'].to_list()
cities = set(cities)
cities

# Ignore month since all in September
month = df.local_time.str[:7]
month = set(month)
month

# Clean weather column
weather_types = df['weather'].to_list()
weather_types = set(weather_types)
weather_types

df['weather'].loc['BreezyandMostlyCloudy']

df[df['weather'].str.match('BreezyandMostlyCloudy')]

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


df['weather'] = df['weather'].apply(lambda w: w.split("and"))
df['weather']

# Check weather with 2 conditions
df.loc[5827, 'weather']



# df.pokedex_id.loc[(df['pokedex_id'] ==34)] = 1
# df.pokedex_id.loc[(df['pokedex_id'] !=34)] = 0
# df.groupby('pokedex_id')['pokedex_id'].count()
