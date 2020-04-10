"""Copyright 2019, Bruno Aristimunha.

This file is part of paper [Re] Deep Convolution
Neural Network and Autoencoders-Based Unsupervised
Feature Learning of EEG Signals.

--------------------------------------------
Dimension Reduction process.
"""

# Import of scale reduction methods.
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection

from numpy import (
    concatenate,
)

from pandas import (
    DataFrame,
)

from auto_enconder import AutoEnconder

from data_management import (
    save_reduce,
    save_feature_model,
    save_history_model,
)


def reduce_dimension(data, class_,
                     path_dataset,
                     name_type,
                     n_components):
    """Perform dimension reduction and save.

    Parameters
    ----------
    data : array-like  (n_samples, n_features)
        The input data to use in reduce process.
    class_ : array-like  (n_samples, )
        The label data related to the data.
    name_type : str
        With the options: "mae" or "maae".
    n_components : int
        Dimension values to reduce.
    path_dataset : str
        Path where the dataset was saved.
    Returns
    -------
    path_reduced : str
        Path where was saved the reduced dataset.
    """
    if name_type == "pca":
        method = PCA(n_components=n_components,
                     random_state=42)
    else:
        if name_type == "srp":
            method = SparseRandomProjection(n_components=n_components,
                                            random_state=42)
        else:
            raise ValueError("Invalid method option.")

    data_reduced = method.fit_transform(data)

    # Conversion of numpy.array to a DataFrame
    data_reduced = DataFrame(data_reduced)
    # On the DataFrame each column has a numeric value,
    # which is converted to a string
    data_reduced.columns = data_reduced.columns.astype(str)
    # Join with class label
    data_reduced["class"] = class_

    path_reduced = save_reduce(data_reduced=data_reduced,
                               value_encoding_dim=n_components,
                               path_dataset=path_dataset,
                               name_type=name_type)

    return path_reduced


def build_feature(data_train,
                  data_valid,
                  class_train,
                  class_valid,
                  path_dataset,
                  epochs,
                  batch_size,
                  type_loss,
                  value_encoding_dim):
    """Control function to use the AutoEnconder class as a dimension reducer.

    This function also saves the values obtained
    after the dimension reduction process. By performing a process
    of reading and accessing files, this function ended up being
    in the data management file.

    Parameters
    ----------
    data_train : array-like  (n_samples, n_features)
        The input data to use in train process.
    data_valid : array-like  (n_samples, n_features)
        The input data to use in train process.
    class_train : array-like  (n_samples, )
        The label data related to the data_train.
    class_valid : array-like  (n_samples, )
        The label data related to the data_test.

    The rest of the parameters and explanation of the
    variables are homonymous with those of the original class.

    Returns
    -------
    auto_encoder : AutoEnconder class object
        Object of the class already trained.
    path_reduce : str
        Path where the reduced set was saved.reduced
    """
    print("Convert and save with value enconding dimension: {} - {}".format(
        type_loss, value_encoding_dim))

    # Initializing the Auto-Encoder model
    auto_encoder = AutoEnconder(epochs=epochs, batch_size=batch_size,
                                value_encoding_dim=value_encoding_dim,
                                type_loss=type_loss,
                                name_dataset=path_dataset)
    # For validreducedation, as described in the text, we use the test dataset.
    auto_encoder.fit(data_train, data_valid)

    # Data transformation
    data_train_encoded = auto_encoder.transform(data_train)
    data_valid_encoded = auto_encoder.transform(data_valid)

    # Conversion of numpy.array to a DataFrame
    data_reduced = DataFrame(concatenate([data_train_encoded,
                                          data_valid_encoded]))
    # On the DataFrame each column has a numeric value,
    # which is converted to a string
    data_reduced.columns = data_reduced.columns.astype(str)
    # Join with class label
    data_reduced["class"] = concatenate([class_train, class_valid])

    path_reduced = save_reduce(data_reduced=data_reduced,
                               value_encoding_dim=value_encoding_dim,
                               path_dataset=path_dataset,
                               name_type=type_loss)

    save_feature_model(auto_encoder, path_dataset,
                       type_loss, value_encoding_dim)

    save_history_model(auto_encoder, path_dataset,
                       type_loss, value_encoding_dim)

    return auto_encoder, path_reduced
