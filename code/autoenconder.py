import tensorflow.keras.backend as kb
from sklearn.base import BaseEstimator
from tensorflow.keras import losses
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Reshape, UpSampling1D
from tensorflow.keras.models import Model

# TODO: Pq vc esta usando essa funcao de loss? Pq vc define como l2?
# Resposta: Eu estava definindo como segunda perda, relacionada no texto.
# Entendo que não é regularização, vou trocar o nome para evitar confusão.


def MAPE(y_actual, y_pred):
    """Estou inseguro se essa é a mesma perda, se é a reportada foi essa mesmo.
    tudo me leva a crer que ele usou `mean_absolute_error` como primeira perda
    e a segunda perda do artigo foi a `mean_absolute_percentage_error`. Mas 
    gostaria de uma validação. Se for esse cenário, então essa função vai sumir.
    """
    custom_loss = kb.abs((y_actual - y_pred) / kb.mean(y_actual))
    return custom_loss


# TODO: Tente padronizar o codigo no formato PEP8. Classes devem ter CamelCase.
# Resposta: Ok!
class AutoEnconder(BaseEstimator):
    """ Template
    TO-DO: detailed explanation

    Parameters
    ----------




    """

    def __init__(self, epochs=10, batch_size=32,
                 name_dataset=None, value_encoding_dim=2,
                 type_loss='mae'):
        """Template
        TO-DO: detailed explanation

        Parameters
        ----------
        epochs
        batch_size
        name_dataset
        value_encoding_dim
        type_loss
        """
        # auto-enconder parameters
        self._value_encoding_dim = value_encoding_dim
        self._batch_size = batch_size
        self._epochs = epochs
        self._type_loss = type_loss

        # information about when is use
        self._name_dataset = name_dataset

        # information about history train
        self._history = []

        # saving method
        self._method_autoenconder = []
        self._method_enconder = []

    def build_auto_enconder(self):
        """ Template
        TO-DO: detailed explanation

        Parameters
        ----------
        value_encoding_dim : TO-DO


        type_loss: TO-DO


        """
        # TODO: Usually these loss do not use this nomenclature.
        #  L1 and L2 norms are used to refer to regularizers.
        #  Please, consider calling it mean absolute error (mae) e mean squared error (mse)
        #  sendo que o mse eh o mais comum para autoencoders.

        # Done!

        if (self.type_loss == 'mae'):
            fun_loss = losses.mean_absolute_error
        else:
            raise ValueError("Loss function not yet implemented.")

        original_signal = Input(shape=(4096, 1))

        enconded = Conv1D(kernel_size=3, filters=16,
                          padding='same', activation='relu')(original_signal)

        enconded = MaxPooling1D(pool_size=2)(enconded)

        enconded = Conv1D(kernel_size=3, filters=32,
                          padding='same', activation='relu')(enconded)

        enconded = MaxPooling1D(pool_size=2)(enconded)

        enconded = Conv1D(kernel_size=3, filters=64,
                          padding='same', activation='relu')(enconded)

        enconded = MaxPooling1D(pool_size=2)(enconded)

        enconded = Flatten()(enconded)

        enconded = Dense(self.value_encoding_dim, activation='relu')(enconded)

        decoded = Dense(512 * 64, activation='relu', use_bias=False)(enconded)

        decoded = Reshape((512, 64))(decoded)

        decoded = Conv1D(kernel_size=3, filters=64,
                         padding='same', activation='relu')(decoded)
        decoded = UpSampling1D()(decoded)

        decoded = Conv1D(kernel_size=3, filters=32,
                         padding='same', activation='relu')(decoded)

        decoded = UpSampling1D()(decoded)

        decoded = Conv1D(kernel_size=3, filters=16,
                         padding='same', activation='relu')(decoded)
        decoded = UpSampling1D()(decoded)

        decoded = Conv1D(kernel_size=3, filters=1,
                         padding='same', activation='sigmoid')(decoded)

        encoder = Model(original_signal, enconded, name='encoder')

        # TO-DO: Se estiver usando python 3.6+, considere usar f-strings
        #  Senao, procure se habituar com o .format
        #  https://realpython.com/python-f-strings/
        # Resposta: Conheço e estou mudando tudo que tem soma;

        autoencoder = Model(original_signal, decoded,
                            name='autoenconder_{}'.format(self.value_encoding_dim))

        autoencoder.compile(optimizer='adam', loss=fun_loss,
                            metrics=['accuracy'])

        self.method_autoenconder = autoencoder
        self.method_enconder = encoder

    def fit(self, X_train, X_validation):
        """ Template
        TO-DO: detailed explanation

        Parameters
        ----------
        X_train
        X_validation

        """
        # Training auto-enconder
        self.history = self.method_autoenconder.fit(X_train, X_train,
                                                    epochs=self.epochs,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    validation_data=(X_validation, X_validation),
                                                    verbose=0)

    def transform(self, X):
        """ Template
        TO-DO: detailed explanation


        Parameters
        ----------
        X

        """
        return self.method_enconder.predict(X)


def feature_learning(epochs, batch_size, name_dataset,
                     type_loss, value_encoding_dim,
                     X_train, X_test):
    """ Template
    TO-DO: detailed explanation

    Parameters
    ----------
    epochs
    batch_size
    name_dataset
    type_loss
    value_encoding_dim
    X_train
    X_test

    Returns
    -------

    """
    # TODO: procure ser consistente com o tipo de case usando.
    #  Eu sei que o sklearn bagunca um pouco definindo variaveis como X_train e tals.
    #  Mas no restante das suas variaveis seja consistente e de preferencia para o snake_case (PEP8).
    # TODO: Use os underscores de maneira adequada (https://dbader.org/blog/meaning-of-underscores-in-python)
    #  No caso de autoEncoder_ nao havia nenhum naming conflict.
    
    autoEncoder_ = Autoenconder(epochs=epochs,
                                batch_size=batch_size,
                                name_dataset=name_dataset,
                                type_loss=type_loss,
                                value_encoding_dim=value_encoding_dim)

    autoEncoder_.fit(X_train, X_test)

    X_train_encode = autoEncoder_.transform(X_train)
    X_test_encode = autoEncoder_.transform(X_test)

    return X_train_encode, X_test_encode, autoEncoder_