#Data Preprocessing

#Importing libraries
import numpy as np
#Mathematical tools library
import matplotlib.pyplot as plt
#Plot nice charts lib
import pandas as pd
#import and manage datasets lib


#IMPORTING THE DATASET
dataset = pd.read_csv('Data.csv')

#Here we take all independant variables form our dataset
#Taking all the lines and all the columns, except the last one
X = dataset.iloc[:, :-1].values

#Creating dependant variable vector
#Exctracting the dependant values
Y = dataset.iloc[:, 3].values

#TAKING CARE OF MISSING DATA

#(replacing missing data,
# with the mean value of all the other values in that column)

from sklearn.impute import SimpleImputer

simpleImputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
#We anly fit the imputer to the columns, where we have missing data
simpleImputer = simpleImputer.fit(X[:, 1:3])
#Replacing the missing data in the matrix X, with the mean values
X[:, 1:3] = simpleImputer.transform(X[:, 1:3])


# ENCODING CATEGORICAL DATA (Non-numerical data)

# Encoding the Independent Variable


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Constructor params ([name, transformer to use(in that case oneHotEncoder, for dummy encoding), columns to encode], remainder(what to do with other columns))
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = np.array(ct.fit_transform(X), dtype=np.float)


# Encoding Y data

from sklearn.preprocessing import LabelEncoder

#Only using the LanelEncoder, because we only have yes and no, so encoding with 0 and 1, is enough
Y = LabelEncoder().fit_transform(Y)


#SPLITTING THE DATASET into Training set and Test set

#Importing the library
from sklearn.model_selection import train_test_split
#X_train is the training part of the matrix with features
#X_test is the test part of the matrix of features
#Y_train is the training part of the dependant variables, asociated with X_train (same indexes, same observations)
#Y_test is the test part of the dependant variables, asociated with X_test (same indexes, same observations)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#FEATURE SCALING

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
#You must first fit and then transform the training set
X_train = sc_X.fit_transform(X_train)
#We dont need to fit it to the test set, because it is already fitted to the training set
X_test = sc_X.transform(X_test)
#It is discussable if it is necessary to scale the dummy variables
#and it depends on the context, but here we will scale them

