# Polynomial Regression

"""
Business problem :
    That is a model that calcolates (predicts) the salary for job offer
    for new employee.
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#Independant variables (Matrix of features)
# Putting upper bound, because we want X to be seen as a matrix, not a vector
X = dataset.iloc[:, 1:2].values
#Dependant variables (Vector)
y = dataset.iloc[:, 2].values

# We dont split the data into test set and training set, because we need the predictions to be
# very accurate and we need as much data as we have to identify the optimal linear relations in our observations

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
# This will transform our matrix of features into matrix with the independant variable on different powers
# x^0, x^1, x^2, x^3, x^4
polynomial_reg = PolynomialFeatures(degree = 4)
X_polynomial = polynomial_reg.fit_transform(X)
# Creating Multiple Linear Regression with the Polynomial variables matrix
linear_reg_poly = LinearRegression()
linear_reg_poly.fit(X_polynomial, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red') # Real observation points
plt.plot(X, linear_regressor.predict(X), color = 'blue') # Prediction points
plt.title('Linear Regression salary prediction')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results

plt.scatter(X, y, color = 'red') # Real observation points
plt.plot(X, linear_reg_poly.predict(polynomial_reg.fit_transform(X)), color = 'green') # Prediction points
plt.title('Polynomial Regression salary prediction')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Making more continious curve (Taking not only N numbers from 1 to 10, but some Q numbers as well)
# Visualising the more continiuos curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red') # Real observation points
plt.plot(X_grid, linear_reg_poly.predict(polynomial_reg.fit_transform(X_grid)), color = 'purple') # Prediction points
plt.title('Polynomial Regression salary prediction')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
linear_regressor.predict(np.array([6.5]).reshape(1, -1))

# Predicting a new result with Polynomial Regression
linear_reg_poly.predict(polynomial_reg.fit_transform(np.array([6.5]).reshape(1, -1)))

# Think of implementing some functionality that makes deeper comparison for both regression models
