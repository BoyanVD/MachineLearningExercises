#Simple Linear Regression

# Business problem :
#   
#   We have dataset with years of experience of a company employees and
#   their salaries. The problem, that we have to solve, is to find the
#   relation between the salaries and the years of experience.
#   The idea of the model is, that it will be able to find the optimal
#   fitting salary for future employee, based on current employees
#   data.
# 
#   We will do this by implementing simple linear regression model

# Data Preprocessing

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Simple Linear Regression to the Training set
# fomula means that salary is proportional to YearsExperience
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

# Predicting the Test set results
# Vector of predictions
y_test_pred = predict(regressor, newdata = test_set)

# Visualising
# install.packages('ggplot2')
library(ggplot2)

#Visualising the training set results
y_train_pred = predict(regressor, newdata = training_set)
ggplot() + 
  #Plotting all geometric points
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') + 
  #Plotting regression line
  geom_line(aes(x = training_set$YearsExperience, y = y_train_pred),
            colour = 'blue') +
  #Title
  ggtitle('Salary vs Experience (Training set)') + 
  #Naming x axis
  xlab('Years of experience') + 
  #Naming y axis
  ylab('Salary ($)')

# Visualising the test set results
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'green') + 
  geom_line(aes(x = training_set$YearsExperience, y = y_train_pred),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test set)') + 
  xlab('Years of experience') + 
  ylab('Salary ($)')



#Predicting salary for new observation from user input
readdouble <- function()
{
  experience <- readline(prompt="Enter employee years of experience : ")
  return(as.numeric(experience))
}
employeeYearsOfExperience = readdouble()
print(employeeYearsOfExperience)
x <- data.frame(YearsExperience = employeeYearsOfExperience, Salary = 0)
predictedSalary = predict(regressor, newdata = x)
print("Employee's predicted salary, based on other employees is : ")
print(unname(predictedSalary))