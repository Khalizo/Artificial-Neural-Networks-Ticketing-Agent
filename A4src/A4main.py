import numpy as np
import sklearn as sk
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier

# x_axis = [1, 2, 3 ,4 ,5]
# y_axis = [100,200,300,400,500]
# plt.barh(x_axis,y_axis, color ='green', label="label for legend")
# plt.title('The title')
# plt.xlabel("hello")
# plt.ylabel("lol")
# plt.legend(facecolor="gra y", shadow=True, title="title for legend")



# ***************Setting Up The Network***************
# MLPClassifier
solver = 'sgd'  # stochastic gradient descent
learning_rate = 'constant'  # default
learning_rate_init: float  # learning rate
hidden_layer_sizes: tuple  # length = n_layers - 2 #we only need 1 (n_units,)
activation = 'logistic'  # the logistic sigmoid function, others available
momentum: float  # Momentum
verbose = True  # To see the iterations

# a different stopping approach
tol: float  # When the loss or score is not improving by
n_iter_no_change: int  # number of iterations with no change
max_iter: int  # set number of iterations

# hidden units 2
clf = MLPClassifier(solver='sgd',
                    learning_rate_init=0.5, hidden_layer_sizes=(4,),
                    verbose=True, momentum=0.5,
                    activation='logistic',
                    n_iter_no_change=5000, max_iter=20000)


# ***************Encoding The Data***************
# load data set
tickets = pd.read_csv('../tickets.csv')
categories = tickets["Response Team"]
inputs = tickets.loc[:, "Request":"Students"]

# encode the X values
X = array(inputs)
no_bool = X == 'No'
yes_bool = X == 'Yes'
X[no_bool] = 0
X[yes_bool] = 1

# encoding the y values
y = array(pd.get_dummies(categories))

# ***************Split The Data Between Training and Testing***************
# creating training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # training and testing split 80/20
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# fit a backprogation algorithm
clf.fit(X_train, y_train)

# compute the output
predictions = clf.predict(X_test)
proba = clf.predict_proba(X_test)
print(predictions[0:5])
print(proba[0:5])
print(clf.score(X_test, y_test))
joblib.dump(clf, '../mynetwork.joblib')




# ***************Advanced Requirement***************
# fit a linear regression model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions_L = lm.predict(X_test)


## The line / model
# plt.scatter(y_test, proba)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.show()
