import warnings
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
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning

# x_axis = [1, 2, 3 ,4 ,5]
# y_axis = [100,200,300,400,500]
# plt.barh(x_axis,y_axis, color ='green', label="label for legend")
# plt.title('The title')

# plt.legend(facecolor="gra y", shadow=True, title="title for legend")


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

# encoding the y values with a one - hot encoding method
y = array(pd.get_dummies(categories))

# ***************Split The Data Between Training and Testing***************
# creating training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # training and testing split 80/20

# ***************Setting Up The Models***************
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


# setting up MLP params for models with different numbers of hidden units
params = [{'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (1,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (2,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (3,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (4,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (5,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (6,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (7,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (8,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (9,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000},
          {'solver': 'sgd', 'momentum': 0, 'learning_rate_init': 0.5,
           'hidden_layer_sizes': (10,), 'momentum': 0.5, 'activation': 'logistic',
           'n_iter_no_change': 5000, 'max_iter': 20000}]

labels = ["1 hidden unit", "2 hidden units", "3 hidden units", "4 hidden units","5 hidden units",
          "6 hidden units", "7 hidden units", "8 hidden units", "9 hidden units","10 hidden units"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'},
             {'c': 'black', 'linestyle': '--'},
             {'c': 'magenta', 'linestyle': '-'},
             {'c': 'magenta', 'linestyle': '--'}]


# Function for plotting the different models
def plot_on_dataset(X, y, X_test, y_test, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)

    mlps = [] # list of models
    # initiating the models
    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(**param)

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            mlp.fit(X, y)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
        print("Test set score: %f" % mlp.score(X_test, y_test))
    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)


fig, axes = plt.subplots(figsize=(15, 10))
plot_on_dataset(X_train, y_train, X_test, y_test, ax=axes, name='Loss vs No. Epochs For Models With Varying Hidden Units')

fig.legend(axes.get_lines(), labels, ncol=3, loc="best", shadow =True)
plt.xlabel("No. of Epochs")
plt.ylabel("Loss")
plt.show()

# joblib.dump(clf, '../mynetwork.joblib')

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
