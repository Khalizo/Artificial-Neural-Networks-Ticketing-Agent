from sklearn.svm import SVC
from Basic import *
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# converts the outputs into 1D arrays for training.
y_train_adv = [np.where(r == 1)[0][0] for r in y_train]
y_test_adv = [np.where(r == 1)[0][0] for r in y_test]


def run_svc():
    """
    Runs the Support Vector Machine (SVM) Classifier. Trains the model and scores it using the test data set 
    :return: svc
    """""
    svc = SVC(gamma='scale', C=1, verbose=False)
    svc.fit(X_train, y_train_adv)
    print("SVC Test Score: {0}".format(svc.score(X_test, y_test_adv)))
    return svc


def run_neigh():
    """
    Runs the K-Nearest Neighbours (KNN) Classifier. Trains the model and scores it using the test data set 
    :return: neigh
    """""

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train_adv)
    print("KNN Test Score: {0}".format(neigh.score(X_test,y_test_adv)))
    return neigh


def run_dtree():
    """
    Runs the Decision Tree Classifier. Trains the model and scores it using the test data set 
    :return: dtree
    """""
    dtree = DecisionTreeClassifier(random_state=0)
    dtree.fit(X_train, y_train_adv)
    print("Decision Tree Test Score: {0}".format(dtree.score(X_test, y_test_adv)))
    return dtree


def run_naive():
    """
    Runs the Naive Bayes Classifier. Trains the model and scores it using the test data set 
    :return: naive
    """""
    naive = GaussianNB()
    naive.fit(X_train, y_train_adv)
    print("Naive Bayes Test Score: {0}".format(naive.score(X_test, y_test_adv)))
    return naive


# Setting up the confusion matrix
classifiers = [run_svc, run_naive, run_dtree, run_neigh]
titles = ["SVC Confusion Matrix", "Naive Bayes Confusion Matrix", "Decision Tree Confusion Matrix",
          "KNN Confusion Matrix"]
class_names = "Credentials", "Datawarehouse", "Emergencies", "Equipment", "Networking"


def advanced():
    """
    Runs all 4 algorithms and plots their test scores on separate confusion matrices
    :return: 
    """""
    for classifier, title in zip(classifiers, titles):
        disp = plot_confusion_matrix(classifier(), X_test, y_test_adv,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize='true')
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


