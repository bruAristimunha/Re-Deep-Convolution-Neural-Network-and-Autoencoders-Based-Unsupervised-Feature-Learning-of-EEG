"""Copyright 2019, Bruno Aristimunha.

This file is part of paper [Re] Deep Convolution
Neural Network and Autoencoders-Based Unsupervised
Feature Learning of EEG Signals.

--------------------------------------------
Auto-Encoder class and implemented loss function.
"""
from sklearn.base import BaseEstimator
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv1D,
    MaxPooling1D,
    Reshape,
    UpSampling1D,
)

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def mean_absolute_average_error(y_true, y_pred):
    """Reproduction of equation 11 presented in the original article.

    Paper Url: https://ieeexplore.ieee.org/document/8355473#deqn11
    Although the text suggests that the second loss function
    presented is `mean_absolute_percentage_error`,
    there is a divergence in the equation.
    Thus, we chose to reproduce the formula presented, instead
    of the text.
    To this end, we adapted the code available in
    Tensorflow to calculate the "mean_absolute_percentage_error"
    (MAPE), and we call it mean_absolute_average_error (maae).
    TensorFlow Url:
    https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/losses.py#L786-L797

    Parameters
    ----------
    y_true :  array-like
        Ground truth values. shape = `[batch_size, d0, .. dN]`

    y_pred :  array-like
        The predicted values. shape = `[batch_size, d0, .. dN]`

    Returns
    -------
        loss float `Tensor`
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = math_ops.abs((y_true - y_pred) / K.maximum(K.mean(y_true), K.epsilon()))
    return 100.0 * K.mean(diff, axis=-1)


class AutoEnconder(BaseEstimator):
    """AutoEnconder Class.

    Reproduction of the AutoEncoder architecture reported
    in the article by T. Wen and Z. Zhang (2018).

    The class reproduces the design pattern present in the scikit-learn
    library, in addition to inheriting the BaseEstimor class allowing
    compatibility with GridSearch validation methods


    Attributes
    ----------
    epochs : int
        Number of epochs that the architecture will be trained.

    batch_size : int
        Number of examples to be used in each epoch.

    value_encoding_dim : int
        Size of the latent space that architecture will
        learn in the process of decoding and encoding.

    type_loss : str
        Which loss function will be minimized in the learning proces,
        with the options: "mae" or "maae"

    name_dataset : str
        Name of the dataset in which the AutoEncoder
        will learn the latent space. For convenience we use
        Pathname as a name.


    Methods
    -------
    build_auto_enconder(self)
        Function for building and compiling the AutoEnconder class

    fit(self, X_train, X_validation):
        Function to Fit the model to learn how to represent a latent
        space by encoding and decoding the original signal

    transform(self, X):
        Function for transforming the vector with original dimensions
        to latent dimensions.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    # pylint: disable=no-member

    def __init__(
        self,
        epochs=10,
        batch_size=32,
        value_encoding_dim=2,
        type_loss="mae",
        name_dataset=None,
    ):
        """Construct objects intended to AutoEnconder."""
        # auto-enconder parameters
        self.value_encoding_dim = value_encoding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.type_loss = type_loss

        # information about when is use
        self.name_dataset = name_dataset

        # information about history train
        self.learn_history = []

        # saving method
        self.method_autoenconder = []
        self.method_enconder = []

    def build_auto_enconder(self):
        """Build and compile the AutoEnconder class.

        Create a AutoEnconder Model.
        Loss function and the number of dimensions are parameters.
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        input_1 (InputLayer)         [(None, 4096, 1)]         0
        _________________________________________________________________
        conv1d (Conv1D)              (None, 4096, 16)          64
        _________________________________________________________________
        maX_pooling1d (MaXPooling1D) (None, 2048, 16)          0
        _________________________________________________________________
        conv1d_1 (Conv1D)            (None, 2048, 32)          1568
        _________________________________________________________________
        maX_pooling1d_1 (MaXPooling1 (None, 1024, 32)          0
        _________________________________________________________________
        conv1d_2 (Conv1D)            (None, 1024, 64)          6208
        _________________________________________________________________
        maX_pooling1d_2 (MaXPooling1 (None, 512, 64)           0
        _________________________________________________________________
        flatten (Flatten)            (None, 32768)             0
        _________________________________________________________________
        dense (Dense)                (None, 2)                 65538
        _________________________________________________________________
        dense_1 (Dense)              (None, 32768)             65536
        _________________________________________________________________
        reshape (Reshape)            (None, 512, 64)           0
        _________________________________________________________________
        conv1d_3 (Conv1D)            (None, 512, 64)           12352
        _________________________________________________________________
        up_sampling1d (UpSampling1D) (None, 1024, 64)          0
        _________________________________________________________________
        conv1d_4 (Conv1D)            (None, 1024, 32)          6176
        _________________________________________________________________
        up_sampling1d_1 (UpSampling1 (None, 2048, 32)          0
        _________________________________________________________________
        conv1d_5 (Conv1D)            (None, 2048, 16)          1552
        _________________________________________________________________
        up_sampling1d_2 (UpSampling1 (None, 4096, 16)          0
        _________________________________________________________________
        conv1d_6 (Conv1D)            (None, 4096, 1)           49
        =================================================================
        Total params: 159,043
        Trainable params: 159,043
        Non-trainable params: 0
        """
        if self.type_loss == "mae":
            fun_loss = losses.mean_absolute_error
        elif self.type_loss == "maae":
            fun_loss = mean_absolute_average_error
        elif self.type_loss == "mape":
            fun_loss = losses.mean_absolute_percentage_error
        else:
            raise ValueError("Loss function not yet implemented.")

        original_signal = Input(shape=(4096, 1))

        layer = Conv1D(16, kernel_size=3, padding="same", activation="relu")(
            original_signal
        )
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = Conv1D(32, kernel_size=3, padding="same", activation="relu")(layer)
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = Conv1D(64, kernel_size=3, padding="same", activation="relu")(layer)
        layer = MaxPooling1D(pool_size=2)(layer)
        layer = Flatten()(layer)
        enconded = Dense(self.value_encoding_dim, activation="relu")(layer)

        layer = Dense(512 * 64, activation="relu", use_bias=False)(enconded)
        layer = Reshape((512, 64))(layer)
        layer = Conv1D(64, kernel_size=3, padding="same", activation="relu")(layer)
        layer = UpSampling1D()(layer)
        layer = Conv1D(32, kernel_size=3, padding="same", activation="relu")(layer)
        layer = UpSampling1D()(layer)
        layer = Conv1D(16, kernel_size=3, padding="same", activation="relu")(layer)
        layer = UpSampling1D()(layer)
        decoded = Conv1D(1, kernel_size=3, padding="same", activation="sigmoid")(layer)

        self.method_enconder = Model(original_signal, enconded, name="encoder")

        auto_enconder_name = "autoenconder_m_{}_loss_{}".format(
            self.value_encoding_dim, self.type_loss
        )

        self.method_autoenconder = Model(
            original_signal, decoded, name=auto_enconder_name
        )

        self.method_autoenconder.compile(
            optimizer="adam", loss=fun_loss, metrics=["accuracy"]
        )

    def fit(self, data_train, data_valid):
        """Fit the model to learn.

        Learn how to represent a latent space by
        encoding and decoding the original signal.

        Parameters
        ----------
        data_train : array-like  (n_samples, n_features)
            The input data to use in train process.
        data_validation : array-like of shape (n_samples, n_features)
            The input data to use in validation process

        Returns
        -------
        self : returns a trained AutoEnconder class model.
        """
        self.build_auto_enconder()
        # Training auto-enconder
        history = self.method_autoenconder.fit(
            data_train,
            data_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(data_valid, data_valid),
            verbose=0,
        )
        self.learn_history = history
        return self

    def transform(self, data):
        """Transform the vector.

        From original dimensions to latent dimensions.

        Parameters
        ----------
        data : array-like  (n_samples, n_features)
            The input data to use in transform process.

        Returns
        -------
        _ : array-like  (n_samples, value_encoding_dim)
            The data transformed to latent dimensions format.
        """
        return self.method_enconder.predict(data)
