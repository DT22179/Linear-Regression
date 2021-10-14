import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv')

# Separating Dependent and Independent Features
X = df.drop('Profit', axis = 1)
y = df['Profit']

# Converting Categorical Feature into One HotEncoding
states = pd.get_dummies(X['State'], drop_first=True)
X = X.drop('State', axis=1)
X = pd.concat([X,states],axis=1)

# Splitting dataset into train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predicting test set results.
y_pred = reg.predict(X_test)

# Checking for R Square value
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)



