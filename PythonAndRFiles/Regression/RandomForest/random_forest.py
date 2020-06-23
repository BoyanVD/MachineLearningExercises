# Random forest regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Create your regressor here
from sklearn.ensemble import RandomForestRegressor
numberOfTrees = 300
regressor = RandomForestRegressor(n_estimators = numberOfTrees, random_state = 0)
# Fitting the Random Forest Regression Model to the dataset
regressor.fit(X, y)

# Predicting a new result
valueToPredict = 6.5
y_pred = regressor.predict(np.array([[valueToPredict]]))

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()