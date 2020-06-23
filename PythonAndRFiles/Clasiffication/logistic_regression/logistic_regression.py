#Logistic Regression

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import constants as const
from matplotlib.colors import ListedColormap

# Visualising the Training set results function
def visualize_results(X_train, y_train, set_name):
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap((const.NEGATIVE_COLOR, const.POSITIVE_COLOR)))
    plt.xlim(X1.min(), X1.max())
    plt.xlim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap((const.NEGATIVE_COLOR, const.POSITIVE_COLOR))(i), label = j)
    plt.title(const.REGRESSION_NAME + ' (' + set_name + ')')
    plt.xlabel(const.X_AXIS_NAME)
    plt.ylabel(const.Y_AXIS_NAME)
    plt.legend()
    plt.show()
    
# Function, that takes new input and makes predictions for it
def make_prediction_for_age_and_estimated_salary(regressor, scaler):
    age = int(input('Please enter Age : '))
    estimated_salary = int(input('Please enter Estimated Salary : '))
    matrix =  np.array([[age, estimated_salary]])
    matrix = scaler.transform(matrix)
    prediction = regressor.predict(matrix)
    answer = 'to buy' if prediction[0] == 1 else 'not to buy'
    print('The person is more likely ' + answer + ' the car')

# Function, that prints the stats for the confusion matrix  
def results_stats(confusion_matrix):
    for i in range(2):
        for j in range(2):
            correctness = 'correct' if i == j else 'incorrect'
            answer = 'positive' if j == 1 else 'negative'
            print("There are ", confusion_matrix[i][j], " ", correctness, " ", answer, " answers")
    
#Imprting dataset
dataset = pd.read_csv(const.DATASET_NAME)
X = dataset.iloc[:, [const.AGE_INDEX, const.ESTIMATED_SALARY_INDEX]].values
y = dataset.iloc[:, const.PURCHASED_INDEX]

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = const.TEST_SIZE, random_state = const.RANDOM_STATE)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Linear Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = const.RANDOM_STATE)
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
results_stats(confusion_matrix)

# Visualising the Training set results
visualize_results(X_train, y_train, 'Training Set')

# Make prediction
make_prediction_for_age_and_estimated_salary(classifier, sc_X)


"""
    The mistakes of the model are due to the fact that the classifier is linear,
    but the observations are not linerly distributed.
"""