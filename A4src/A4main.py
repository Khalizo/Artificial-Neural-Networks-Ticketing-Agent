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
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from A4src.configuration import *

# ***************Encoding The Data***************
# load data set
tickets = pd.read_csv('../data/tickets.csv')
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
tol: float  # When the loss or score is not improving by
n_iter_no_change: int  # number of iterations with no change
max_iter: int  # set number of iterations

# initialise MLP classifier
clf = MLPClassifier(solver='sgd',
                    learning_rate_init=0.5, hidden_layer_sizes=(5,),
                    verbose=False, momentum=0.3, tol=0.00001,
                    activation='logistic',
                    n_iter_no_change=10, max_iter=10000)

# Implementing Grid Search to find the best learning rate and momentum
param_grid = [{'solver': ['sgd'], 'momentum': [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               'learning_rate_init': [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]


def grid_search(param_grid):
    grid_search_fit = GridSearchCV(estimator=clf,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=1)
    grid_search_fit = grid_search_fit.fit(X_train, y_train)
    grid_csv = pd.DataFrame(grid_search_fit.cv_results_).to_csv('../data/grid_search_results.csv')
    print(grid_search.best_params_)

# function for exporting to CSV
def export_to_csv(model_name, train_score, train_loss, test_score, iterations, filename):
    results = pd.DataFrame({'Model Name': model_name, 'Training Set Score': train_score,
                            'Training Set Loss': train_loss,
                            'Test Set Score': test_score, 'Iterations': iterations})
    results.to_csv(filename)

# Function for plotting the different models
def plot_on_models(X, y, X_test, y_test, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)

    mlps = []  # list of models
    train_scores = []
    train_losses = []
    test_scores = []
    iterations = []

    # initiating the models
    for label, param, saved_network in zip(labels, params, saved_networks):
        print("training: %s" % label)
        mlp = MLPClassifier(solver='sgd', learning_rate_init=0.8, momentum=0.1, activation='logistic',
                            tol=0.00001, n_iter_no_change=10, max_iter=20000, **param)

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            mlp.fit(X, y)
            joblib.dump(mlp, saved_network)
        # add models and data points to their respective arrays
        mlps.append(mlp)
        train_scores.append(mlp.score(X, y))
        train_losses.append(mlp.loss_)
        test_scores.append(mlp.score(X_test, y_test))
        iterations.append(mlp.n_iter_)

        print("Training set accuracy: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
        print("Test set accuracy: %f" % mlp.score(X_test, y_test))
        print("Iterations: %f" % mlp.n_iter_)

    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)

    export_to_csv(labels, train_scores, train_losses, test_scores, iterations, model_results)


# ***************Set up plot***************
fig, axes = plt.subplots(figsize=(15, 10))
plot_on_models(X_train, y_train, X_test, y_test, ax=axes,
               name='Loss vs No. Epochs For Models With Varying Hidden Units')
fig.legend(axes.get_lines(), labels, ncol=3, loc="best", shadow=True)
plt.xlabel("No. of Epochs")
plt.ylabel("Loss")
plt.show()
plt.savefig(save_fig)



# # fit a linear regression model
# lm = linear_model.LinearRegression()
# model = lm.fit(X_train, y_train)
# predictions_L = lm.predict(X_test)

## The line / model
# plt.scatter(y_test, proba)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.show()
# x_axis = [1, 2, 3 ,4 ,5]
# y_axis = [100,200,300,400,500]
# plt.barh(x_axis,y_axis, color ='green', label="label for legend")
# plt.title('The title')

# plt.legend(facecolor="gra y", shadow=True, title="title for legend")
