#Data Preprocessing

#Importing dataset
dataset = read.csv('Data.csv')

#Taking care of missing data

#Taking the column age of the dataset
#in ifelse we put condition, what happens if true, and what happens if false
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)
#In ave we have put the function, that fills the missing value, with the mean value of all other values
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

#Encoding categorical data
#In R we are going to use the factor function,
#which transforms our categorical variables into
#numeric categories, but it will see the variables as factors
# You will be able to choose the labels of those factors

#We just transform the categorical column into
# a column of factors, specifiyng what the factors are

#c -> vector in R
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))

#Now we do the same for the purchased column
dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No', 'Yes'),
                         labels = c(0, 1))



#SPLITTING THE DATASET INTO TRAIN SET AND TEST SET

#MUST run this scripts to install and activate the library
#install.packages('caTools')
#library(caTools)

set.seed(123)
#This method will return true if the observation is chosen to go to the training set,
#and it will return false, if it is chosen to go to the test set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#FEATURE SCALING

#Here Country and Purchased are factors, not numerical values
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])