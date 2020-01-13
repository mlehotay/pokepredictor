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
from sklearn.neighbors import KNeighborsClassifier

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
# weather_types = df['weather'].to_list()
# weather_types = set(weather_types)
# weather_types
#
# df['weather'].loc['BreezyandMostlyCloudy']
#
# df[df['weather'].str.match('BreezyandMostlyCloudy')]
#
# weather_types = {
# 'Breezy',
# 'Clear',
# 'DangerouslyWindy',
# 'Drizzle',
# 'Dry',
# 'Foggy',
# 'HeavyRain',
# 'Humid',
# 'LightRain',
# 'MostlyCloudy',
# 'Overcast',
# 'PartlyCloudy',
# 'Rain',
# 'RainandWindy',
# 'Windy'
# }
#
#
# df['weather'] = df['weather'].apply(lambda w: w.split("and"))
# df['weather']
#
# # Check weather with 2 conditions
# df.loc[5827, 'weather']
#
# df['new'] = (df['pokedex_id'] == 34)
# df['new'].sum()


###############



# Pikachu
# y_new = [[1 if v ==34  else 0 for v in df['pokedex_id']]]
#
# y_new
#
# y_newdf = pd.DataFrame(y_new) ### convert list into dF
#
# y_newdf.isnull().sum()
#
# y_pikachu = y_newdf.T
#
# y_pikachu
#
# y_pikachu[0].sum()





# X= df.drop('pokedex_id', axis = 1)

# X.info()


# target

# Model

# target = df['pokedex_id']
y = df['pokedex_id']
X = df.drop(['pokedex_id', 'weather', 'local_time'], axis = 1)
l = len(y)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

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

# model = KNeighborsClassifier()
model = LinearRegression()
model.fit(Z_train, y_train)

# catboost_model = CatBoostRegressor(iterations=20, depth=2, loss_function="RMSE",verbose=False)
# catboost_model.fit(train_pool, eval_set=test_pool)

# Pipe
pipe = make_pipeline(mapper, model)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

#Pickle
pickle.dump(pipe, open('model/pipe.pkl', 'wb'))
del pipe
pipe = pickle.load(open('model/pipe.pkl', 'rb'))
pipe
