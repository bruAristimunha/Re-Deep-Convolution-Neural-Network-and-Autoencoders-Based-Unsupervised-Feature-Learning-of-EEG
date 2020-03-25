"""
Description
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
)


def reduce_dimension(X, y,
                     path_dataset,
                     name_type,
                     n_components):
    """
    TODO
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

    X_reduced = method.fit_transform(X)

    # Conversion of numpy.array to a DataFrame
    data_reduced = DataFrame(X_reduced)
    # On the DataFrame each column has a numeric value,
    # which is converted to a string
    data_reduced.columns = data_reduced.columns.astype(str)
    # Join with class label
    data_reduced["class"] = y

    path_reduced = save_reduce(data_reduced=data_reduced,
                               value_encoding_dim=n_components,
                               path_dataset=path_dataset,
                               name_type=name_type)

    return path_reduced


def build_feature(X_train,
                  X_valid,
                  y_train,
                  y_valid,
                  path_dataset,
                  epochs,
                  batch_size,
                  type_loss,
                  value_encoding_dim):
    """
    Control function to use the AutoEnconder
    class as a dimension reducer.

    This function also saves the values obtained
    after the dimension reduction process. By performing a process
    of reading and accessing files, this function ended up being
    in the data management file.

    Parameters
    ----------
    X_train : array-like  (n_samples, n_features)
        The input data to use in train process.

    X_valid : array-like  (n_samples, n_features)
        The input data to use in train process.

    Y_train : array-like  (n_samples, )
        The label data related to the X_train.

    Y_valid : array-like  (n_samples, )
        The label data related to the X_test.

    The rest of the parameters and explanation of the
    variables are homonymous with those of the original class.

    Returns
    -------
    auto_encoder : AutoEnconder class object
        Object of the class already trained.

    path_reduce : str
        Path where the reduced set was saved.reduced

    """

    print("Convert and save with value enconding dimension: {}".format(
        value_encoding_dim))

    # Initializing the Auto-Encoder model
    auto_encoder = AutoEnconder(epochs=epochs, batch_size=batch_size,
                                value_encoding_dim=value_encoding_dim,
                                type_loss=type_loss,
                                name_dataset=path_dataset)
    # For validreducedation, as described in the text, we use the test dataset.
    auto_encoder.fit(X_train, X_valid)

    # Data transformation
    X_train_encoded = auto_encoder.transform(X_train)
    X_valid_encoded = auto_encoder.transform(X_valid)

    # Conversion of numpy.array to a DataFrame
    data_reduced = DataFrame(concatenate([X_train_encoded, X_valid_encoded]))
    # On the DataFrame each column has a numeric value,
    # which is converted to a string
    data_reduced.columns = data_reduced.columns.astype(str)
    # Join with class label
    data_reduced["class"] = concatenate([y_train, y_valid])

    path_reduced = save_reduce(data_reduced=data_reduced,
                               value_encoding_dim=value_encoding_dim,
                               path_dataset=path_dataset,
                               name_type=type_loss)

    save_feature_model(auto_encoder, path_dataset,
                       type_loss, value_encoding_dim)

    return auto_encoder, path_reduced
