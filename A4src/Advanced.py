from sklearn.svm import SVC
from A4src.Basic import *
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

y_train_adv = [np.where(r == 1)[0][0] for r in y_train]
y_test_adv = [np.where(r == 1)[0][0] for r in y_test]
class_names = "Credentials", "Datawarehouse", "Emergencies", "Equipment", "Networking"


def run_svc():
        svc = SVC(gamma='scale', C=1, verbose=False)
        svc.fit(X_train, y_train_adv)
        print(svc.score(X_test,y_test_adv))
        return svc


def run_neigh():
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train_adv)
    print(neigh.score(X_test,y_test_adv))
    return neigh


def run_gpc():
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state = 0).fit(X_train, y_train_adv)
    print(gpc.score(X_test, y_test_adv))
    return gpc


def run_dtree():
    dtree = DecisionTreeClassifier(random_state=0)
    dtree.fit(X_train, y_train_adv)
    print(dtree.score(X_test, y_test_adv))
    return dtree


def run_forest():
    forest = RandomForestClassifier(max_depth=2, random_state=0)
    forest.fit(X_train, y_train_adv)
    print(forest.score(X_test, y_test_adv))
    return forest


def confusion_matrix():
        title ='Random Forest Confusion Matrix'
        disp = plot_confusion_matrix(run_forest(), X_test, y_test_adv,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues, normalize='true')
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

        plt.show()

confusion_matrix()
