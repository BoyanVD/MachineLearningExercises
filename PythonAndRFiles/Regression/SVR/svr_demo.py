# Support Vector Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data preprocessing
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling(SVR doesn't apply the feature scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# Creating the regressor
from sklearn.svm import SVR
# The kernel is what type of SVR do you want (Linear, Polynomial, Gassion)
regressor = SVR(kernel = 'rbf')

# Fitting the SVR to the dataset
regressor.fit(X, y)

# Predicting a new result
valueToPredict = 6.5
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[valueToPredict]]))))

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

"""
The CEO in that case is considered as an
outlyer. The SVR model is having some penalty
parameters, selected in the algorithm, and because
the CEO is too far away from the other datapoints, so that
is why it is ignored.
"""