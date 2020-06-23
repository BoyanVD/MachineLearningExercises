
#Multiple linear regression

"""
Business problem :
    We have 50 companies with 5 properties for each one.
    We have to analyse the dataset and create a model, which will tell
    the venture capitalist which companies are worth investing in. We
    have to tell which properties are most influent on profit, based on
    the dataset, as well.
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
#Independant variables (Matrix of features)
X = dataset.iloc[:, :-1].values
#Dependant variables (Vector)
y = dataset.iloc[:, 4].values

# ENCODING CATEGORICAL DATA (Non-numerical data)
# Dummy encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#Avoiding the dummy variable Trap (Removing the first column)
X = X[:, 1:]

# Building the optimal model using Backward Elimination
# 1. Backward Elimination preparation
import statsmodels.api as sm
# Add column of 1-ones, presenting the x0 coeff on b0 Constant (Simulation of the Constant)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# 2. Backward Elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        #Finding the max p-value
        maxVar = max(regressor_OLS.pvalues).astype(float)
        #If there is p-value, greater than the Significance Level, we will remove it
        if maxVar > sl:
            #Searching for the independant variable, which p-value is the highest
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    #Deleting the dependant variable, with highest p-value, greater than SL
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

#Choosing Significance Level
SL = 0.05
#Creating the first optimal matrix, which inculdes all values in the beginning
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
y_test_pred = regressor.predict(X_test)

# Creating a vector, that shows the differences between real and predicted values 
y_comparison = y_test - y_test_pred

# Function, that takes new input and makes predictions for it
def makePredictionForNewRDSpendValue():
    RDSpend = float(input('Please enter amount of R&D Spend in $ : '))
    RDSpendAsArray = np.array([RDSpend])
    RDSpendAsArray = np.vstack(([1], RDSpendAsArray)).T
    predictedProfit = regressor.predict(RDSpendAsArray)
    print('The predicted profit for this Startup is : ', predictedProfit[0], '$')

makePredictionForNewRDSpendValue()