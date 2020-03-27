"""
  TO-DO:  Description
"""

from pathlib import Path
from os.path import join

from pandas import read_parquet, DataFrame, concat

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

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
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def methods_classification(n_neighbors=3,
                           kernel_a="linear", kernel_b="rbf", max_depth=5,
                           n_estimators=10, random_state=42, max_features=1):
    """
    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to use
    kernel_a : str
        Specifies the kernel type to be used in the SVC a.
    kernel_b : str
        Specifies the kernel type to be used in the SVC b.
    max_depth : int
        The maximum depth of the tree.
    n_estimators : int
        The number of trees in the forest.
    random_state : int
        Controls the randomness
    max_features : int
        The number of features to consider when looking for the best split.

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


def read_feature_data(base_fold, dim):
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
    name_reduced = join(base_fold, "reduced_dataset_{}.parquet".format(dim))
    X = read_parquet(name_reduced, engine="pyarrow").drop(["class"], 1)
    y = read_parquet(name_reduced, engine="pyarrow")["class"]

    return X, y


def save_classification(scores, base_fold):
    """
    TODO
    """
    fold = Path(base_fold) / "save"

    if not fold.exists():
        fold.mkdir(parents=True, exist_ok=True)
    print(scores)


@ignore_warnings(category=ConvergenceWarning)
def run_classification(path_dataset,
                       name_type,
                       range_values):
    """
    TO-DO
    """

    path_base = join(path_dataset, "reduced")

    if name_type in ('mae', 'maae'):
        path_read = join(path_base, "ae_{}".format(name_type))
    else:
        path_read = join(path_base, name_type)

    scores = []

    files = [read_feature_data(path_read, dim) for dim in range_values]

    classifiers = methods_classification(n_neighbors=3,
                                         kernel_a="linear",
                                         kernel_b="rbf",
                                         max_depth=5,
                                         n_estimators=10,
                                         random_state=42,
                                         max_features=1)

    for ind, dim in enumerate(range_values):

        print("Running with {} dimensions".format(dim))

        X = files[ind][0]
        y = files[ind][1]

        for name_classifier, classifier in classifiers:

            scoring = ["accuracy"]  # , "precision", "recall","f1", "roc_auc"]

            #The following clf uses minmax scaling
            clf = make_pipeline(MinMaxScaler(), classifier)

            score = cross_validate(clf, X, y, cv=5, scoring=scoring)
            # Aggregate name in cross_validate

            score.update({"name_classifier": name_classifier,
                          "Dimension": dim,
                          "name_type": name_type})

            scores.append(DataFrame(score))

    scores = concat(scores).reset_index()

    scores.columns = ["5-fold"] + scores.columns[1:].tolist()

    scores["5-fold"] = scores["5-fold"] + 1

    return scores
