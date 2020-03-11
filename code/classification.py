"""
  TO-DO:  Description
"""

from pathlib import Path
from os.path import join

from pandas import read_parquet, DataFrame, concat
from imblearn.metrics import specificity_score

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
# Classification methods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)


def methods_classification(n_neighbors=3,
                           kernel_a="linear", kernel_b="rbf", max_depth=5,
                           n_estimators=10, random_state=42, max_features=1):
    """
    Parameters
    ----------
    n_neighbors : int

    kernel_a : str

    kernel_b : str

    max_depth : int

    n_estimators : int

    random_state : int

    max_features : int

    Returns
    -------
    classifiers : list
    """

    k_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)  # 1
    svm_linear = svm.SVC(kernel=kernel_a)  # 2
    svm_radial = svm.SVC(kernel=kernel_b)  # 3
    decision_tree = DecisionTreeClassifier(max_depth=max_depth)  # 4
    random_forest = RandomForestClassifier(n_estimators=n_estimators,
                                           random_state=random_state,
                                           max_features=max_features)  # 5
    multi_layer = MLPClassifier()  # 6
    ada_boost = AdaBoostClassifier(random_state=random_state)  # 7
    gaussian_nb = GaussianNB()

    ensemble = VotingClassifier(estimators=[("k-NN", k_neighbors),
                                            ("SVM1", svm_linear),
                                            ("SVM2", svm_radial),
                                            ("DT", decision_tree),
                                            ("RF", random_forest),
                                            ("MLP", multi_layer),
                                            ("ADB", ada_boost),
                                            ("GNB", gaussian_nb)], voting="hard")

    classifiers = [("k_neighbors", k_neighbors),
                   ("svm_linear", svm_linear),
                   ("svm_radial", svm_radial),
                   ("decision_tree", decision_tree),
                   ("random_forest", random_forest),
                   ("multi_layer", multi_layer),
                   ("ada_boost", ada_boost),
                   ("gaussian_nb", gaussian_nb),
                   ("ensemble", ensemble)]

    return classifiers


def read_feature_data(base_fold, dim, type_loss):
    """



    Parameters
    ----------
    base_fold : str
        Pathname to indicate where to download the dataset.  

    dim : int
        Size of the latent space that architecture will
        learn in the process of decoding and encoding.

    type_loss : str
        Which loss function will be minimized in the learning proces,
        with the options: "mae" or "maae"

    Returns
    -------
    X : array

    y : array

    """
    name_train = join(base_fold, "train_{}_{}.parquet".format(dim, type_loss))
    X_train = read_parquet(name_train, engine="pyarrow").drop(["class"], 1)
    y_train = read_parquet(name_train, engine="pyarrow")["class"]

    name_test = join(base_fold, "test_{}_{}.parquet".format(dim, type_loss))

    X_test = read_parquet(name_test, engine="pyarrow").drop(["class"], 1)
    y_test = read_parquet(name_test, engine="pyarrow")["class"]

    X = X_train.append(X_test)
    y = y_train.append(y_test)

    return X, y


def save_classification(scores, base_fold):
    """


    """
    path_save = join(base_fold, "save")

    if not fold.exists():
        fold.mkdir(parents=True, exist_ok=True)


        
def run_classification(base_fold,
                       type_loss,
                       range_values):
    """


    """

    path_read = join(base_fold, "feature_learning")

    scores = []
    
    
    classifiers = methods_classification(n_neighbors=3, 
                                     kernel_a='linear', 
                                     kernel_b='rbf', 
                                     max_depth=5,
                                     n_estimators=10, 
                                     random_state=42, 
                                     max_features=1)
    for _, classifier in classifiers:

        for dim in range_values:

            X, y = read_feature_data(path_read, dim, type_loss)

            scoring = ['accuracy']  # , 'precision', 'recall','f1', 'roc_auc']

            score = cross_validate(classifier, X, y, cv=5, scoring=scoring)
            # Aggregate name in cross_validate
            #import pdb; pdb.set_trace()
            score.update({'name_classifier': type(classifier).__name__,
                          'n_dimensions': dim,
                          'type_loss': type_loss})

            scores.append(DataFrame(score))

    scores = concat(scores).reset_index()

    scores.columns = ['n_CV'] + scores.columns[1:].tolist()

    scores['n_CV'] = scores['n_CV'] + 1

    return scores