import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import pickle



data = '../Data/cleaned_data_eda.csv'

df = pd.read_csv(data)

# Checking for null values
df.isnull().sum()



# Checking to see the percentage of null values out of the entire dataset
100 * df.isnull().sum() / len(df)


# The missing amount for year_made is 43%, and the missing amount for made_in is 66%. Since these are such large portions of the dataset and the only way to find that data would be to search for it, and since I don't have accurate reads on the year_made feature - since they're listed as decades I'm going to remove both columns from the dataset
df.drop(['Unnamed: 0', 'year_made', 'made_in'], axis=1, inplace=True)


# The url column also isn't relevant to any model, so I will remove it.
df.drop('urls', axis=1, inplace=True)


# Looking at the synth_types that are null
print(len(df[df['synth_types'].isnull()]))
df[df['synth_types'].isnull()]

re.findall('Nord', df['Description'][133])



# Building a function to try to pull synth types from the description
def synth_type_identifier(row):
    for synth_type in df.synth_types.unique():
        if re.search(str(synth_type).lower(), str(row).lower()):
            return synth_type
        else:
            pass

df['synth_types'].fillna(df['Description'].apply(synth_type_identifier), inplace=True)

df.synth_types.isnull().sum()

# I will remove all remaining null values for synth type
df.dropna(subset=['synth_types'], axis=0, inplace=True)

df.isnull().sum()

df.info()


## ## 2. Feature Engineering


# Converting Condition and synth_types to dummy variables
df = pd.get_dummies(df, columns=['Condition', 'synth_types'], drop_first=True)

# Converting n_keys to numeric
df['n_keys'] = df.n_keys.apply(lambda x: pd.to_numeric(x.split()[0]))

# Creating a new dataframe to drop Description 
new_df = df.drop(['Description', 'Model', 'Brand'], axis=1)

# # 4. Model Building

# First I will make dummy variables out of all the brands in the data set

brand_dummies = pd.get_dummies(df['Brand'], drop_first=True)
new_df = pd.concat([new_df, brand_dummies], axis=1)

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = new_df.drop('Price', axis=1)
y = new_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

lasso_regression = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

# Using GridSearch to find the optimal parameter of alpha for the lasso model with 5-Fold Cross Validation
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

lasso_regressor.fit(X_train_scaled, y_train)

# Printing the best alpha as per the GridSearchCV
print(lasso_regressor.best_params_)

# Printing the best neg MSE score per the GridSearch CV
print(lasso_regressor.best_score_)

lasso = Lasso(alpha=1)

lasso.fit(X_train_scaled, y_train)

lasso_y_pred = lasso.predict(X_test_scaled)

pickle.dump(lasso, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))