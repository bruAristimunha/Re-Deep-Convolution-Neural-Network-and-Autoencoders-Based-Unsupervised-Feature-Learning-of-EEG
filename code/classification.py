"""Copyright 2019, Bruno Aristimunha.

This file is part of paper [Re] Deep Convolution
Neural Network and Autoencoders-Based Unsupervised
Feature Learning of EEG Signals.

--------------------------------------------
Classification methods and function control of process.
"""

from os.path import join

from pandas import DataFrame, concat

from sklearn.model_selection import (
    cross_validate,
    KFold,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
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
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from data_management import (
    read_feature_data,
    save_classification,
)


def methods_classification(n_neighbors=3,
                           kernel_a="linear", kernel_b="rbf", max_depth=5,
                           n_estimators=10, random_state=42, max_features=1):
    """
    Group methods used in classification, as well as creating the ensemble.

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
                                            ("GNB", gaussian_nb)],
                                voting="hard")

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


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UndefinedMetricWarning)
def run_classification(path_dataset,
                       name_type,
                       range_values,
                       cross_values=5):
    """Perform the classification, save the result for all classifiers."""
    scoring = {"accuracy": make_scorer(accuracy_score),
               "precision": make_scorer(precision_score),
               "specificity": make_scorer(recall_score, pos_label=0),
               "sensitivity": make_scorer(recall_score),
               "f-measure": make_scorer(f1_score),
               "roc-auc": make_scorer(roc_auc_score)}

    print("Perform classification on data reduced by : {}".format(name_type))

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

        data = files[ind][0]
        class_ = files[ind][1]

        for name_classifier, classifier in classifiers:

            # The following clf uses minmax scaling
            clf = make_pipeline(MinMaxScaler(), classifier)

            score = cross_validate(clf, data,
                                   class_,
                                   cv=cross_values, scoring=scoring)
            # Aggregate name in cross_validate

            score.update({"name_classifier": name_classifier,
                          "Dimension": dim,
                          "name_type": name_type})

            scores.append(DataFrame(score))

    scores = concat(scores).reset_index()

    columns_name = scores.columns[1:].tolist()

    fold_name_columns = "{}-fold".format(cross_values)

    scores.columns = fold_name_columns + columns_name

    scores[fold_name_columns] = scores[fold_name_columns] + 1

    save_classification(scores, path_dataset, name_type, cross_values)

    return scores


def run_classification_nn(path_dataset, name_type,
                          dim, cross_values,
                          epochs=100):
    """Draft classification function with NN2. Development stopped."""
    path_base = join(path_dataset, "reduced")

    if name_type in ("mae", "maae"):
        path_read = join(path_base, "ae_{}".format(name_type))
    else:
        path_read = join(path_base, name_type)

    data, class_ = read_feature_data(path_read, dim)

    data = data.to_numpy()

    kfold = KFold(n_splits=cross_values, random_state=42, shuffle=True)

    history_acc = []
    accumulate_acc = []

    def create_model():
        model = Sequential()
        model.add(Dense(22, input_dim=dim, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(1, activation="softmax"))
        model.compile(loss="binary_crossentropy",
                      optimizer="adam", metrics=["accuracy"])
        return model

    for train_index, val_index in kfold.split(data):

        model = KerasClassifier(build_fn=create_model,
                                epochs=epochs,
                                batch_size=128,
                                verbose=0)

        history = model.fit(data[train_index], class_[train_index])
        pred = model.predict(data[val_index])

        accuracy = accuracy_score(class_[val_index], pred)
        accumulate_acc.append(accuracy)

        history_acc.append(history.history['loss'])

    return history_acc, accumulate_acc
