"""
Business problem :
    
    We have dataset with years of experience of a company employees and
    their salaries. The problem, that we have to solve, is to find the
    relation between the salaries and the years of experience.
    The idea of the model is, that it will be able to find the optimal
    fitting salary for future employee, based on current employees
    data.
    
    We will do this by implementing simple linear regression model
    
"""

#Simple Linear Regression

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#Independant variables (Matrix of features)
X = dataset.iloc[:, :-1].values
#Dependant variables (Vector)
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
#Vector of predicted values for dependant valiable (predicted salary for the observations in test set)
y_test_pred = regressor.predict(X_test)

#Visualising the Training set results
#Plotting the real observations points (employees)￼
plt.scatter(X_train, y_train, color = 'red')
#Here we put prediction of the train set, because we are drawing the regression line, which is made out of the predicted values, not the real ones
y_train_pred = regressor.predict(X_train)
plt.plot(X_train, y_train_pred, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()

#Visualising the Test set results
#Plotting the real observations points (employees)￼
plt.scatter(X_test, y_test, color = 'green')
#Here it is the same like in train set visualisation,
#because the regression line is built, based on training set and we want to use the same regression line here
plt.plot(X_train, y_train_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()

# Taking user input for experience and making prediction for salary
experience = float(input('Please enter years of working experience : '))
print('The employee optimal predicted salary based on other employees is : ')
predicted_salary =regressor.predict(np.array([experience]).reshape(1, -1))
print(predicted_salary)
