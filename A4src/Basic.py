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
from Config import *

# ***************Encoding The Data***************
# loads data set
tickets = pd.read_csv('../data/tickets.csv')
input_values = tickets.loc[:, "Request":"Students"]


def encode(dataset):
    """
    Encodes the data set using the one-hot encoding method
    :return: X_y
    """""
    categories = dataset["Response Team"]  # separate the output values
    inputs = dataset.loc[:, "Request":"Students"]  # separate the input values
    X_y = {}  # dictionary for storing the inputs and outputs separately

    # encode the inputs values using numpy boolean indexing
    X = array(inputs)
    no_bool = X == 'No'
    yes_bool = X == 'Yes'
    X[no_bool] = 0
    X[yes_bool] = 1
    X_y[0] = X

    # encode the categories
    y = array(pd.get_dummies(categories))
    X_y[1] = y
    return X_y


X = encode(tickets).get(0)
y = encode(tickets).get(1)

# creating training and test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # training and testing split 80/20

# ***************Grid Search Setup***************

# initialise MLP classifier for Implementing GridSearch
clf = MLPClassifier(solver='sgd',
                    learning_rate_init=0.5, hidden_layer_sizes=(5,),
                    verbose=False, momentum=0.3, tol=0.00001,
                    activation='logistic',
                    n_iter_no_change=10, max_iter=10000)

# Implementing Grid Search to find the best learning rate and momentum
param_grid = [{'solver': ['sgd'], 'momentum': [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               'learning_rate_init': [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]


def grid_search(param_grid):
    """
    Runs grid search and exports data into a csv
    :return: 
    """""
    grid_search_fit = GridSearchCV(estimator=clf,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=10,
                                   n_jobs=1)
    grid_search_fit = grid_search_fit.fit(X_train, y_train)
    grid_csv = pd.DataFrame(grid_search_fit.cv_results_).to_csv('../results/grid_search_results.csv')


def export_to_csv(model_name, train_score, train_loss, test_score, iterations, filename):
    """
    Exports data to csv after running the models with varying hidding units
    :return: 
    """""
    results = pd.DataFrame({'Model Name': model_name, 'Training Set Score': train_score,
                            'Training Set Loss': train_loss,
                            'Test Set Score': test_score, 'Iterations': iterations})
    results.to_csv(filename)


def plot_on_models(X, y, X_test, y_test, ax, name):
    """
    Trains 10 different models, provides settings for plotting and exports their performance on the test & training
    sets to a csv
    :return: 
    """""
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

        # in case some parameter combinations don't converge
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


def basic():
    """
    Plots the models for basic
    :return: 
    """""
    fig, axes = plt.subplots(figsize=(10, 10))
    plot_on_models(X_train, y_train, X_test, y_test, ax=axes,
                   name='Loss vs No. Epochs For Models With Varying Hidden Units')
    fig.legend(axes.get_lines(), labels, ncol=3, loc="right", shadow=True)
    plt.xlabel("No. of Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig(save_fig)

