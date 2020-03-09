"""
TODO: Description about the file.


"""

from os.path import join
from pathlib import Path
from zipfile import ZipFile

from bs4 import BeautifulSoup
from numpy import zeros, ones, concatenate, array, reshape, isin
from pandas import DataFrame
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from wget import download

from autoenconder import AutoEnconder


def zip_with_unique(base, list_suffix):
    """ Auxiliary function to generate a paired
    list considering a unique element.

    An adaptation of the convolution function (`zip`)
    to map a single value in a sequence tuple.
    This mapping is a surjective-only.

    An adaptation of the "scalar product"
    to make a zip with vectors of different sizes.
    The first vector must have 1 item,
    while the second must-have n items.

    Parameters
    ----------
    base: array-like, shape (1, )
        One (1) base prefix that will be paired with the suffixes.

    list_suffix : array-like, shape (n_suffix,)
        A suffix list that will be paired with one prefix.

    Returns
    -------
    list: array-like, shape (n_suffix)
        Base added with the suffix.
    """

    return list(base + suffix for suffix in list_suffix)


def download_bonn(path_data="data/boon/") -> [str]:
    """
    Adapted from mne-tools.github.io/mne-features/auto_examples/plot_seizure_example.html
    Code changes were:
        * Adding more folders;
        * Control for folder creation;

    Parameters
    ----------
    path_data : str,
        Path to indicate where to download the data set.

    Returns
    -------
    path_child_fold : str,
        List of strings with the path where
        the dataset files were downloaded.

    """
    fold = Path(path_data)
    child_fold = ["setA", "setB", "setC", "setD", "setE"]
    base_url = "http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/"
    urls_suffix = ["Z.zip", "O.zip", "N.zip", "F.zip", "S.zip"]

    path_child_fold = zip_with_unique(path_data, child_fold)

    if fold.exists():
        print("Folder already exists")
        check_child_folders = [Path(child).exists()
                               for child in path_child_fold]

        if all(check_child_folders):
            print("Subfolders already exist")

            return path_child_fold
    else:
        print("Creating folder")
        # Create parent directory
        fold.mkdir(parents=True, exist_ok=True)
        # This way, the child directory will also be created.
        for child in path_child_fold:
            Path(child).mkdir(parents=True, exist_ok=True)

        urls = zip_with_unique(base_url, urls_suffix)

        print("Downloading and unzipping the files")

        for url, path in list(zip(urls, path_child_fold)):
            file_directory = download(url, path)

            with ZipFile(file_directory, "r") as zip_ref:
                zip_ref.extractall(path)

    return path_child_fold


def download_item(url_base, name_base, page=True):
    """
    Function to download the files in an isolated way.
    Used when the file listing is different from a folder.

    Parameters
    ----------
    url_base : str,
        Url to indicate where to download the dataset.

    name_base : str
        Pathname to indicate where to download the dataset.

    page : bool
        Parameter to be used to download the page's html.

    Returns
    -------
    list : list [str]
        Filter list with files and folder, not included when id people > 10.

    """
    download(url_base, name_base)
    if page:
        base = open(name_base, "r").read()
        soup = BeautifulSoup(base, "html.parser")
        return filter_list([link.get("href") for link in soup.find_all("a")])

    return None


def filter_list(folders_description):
    """
        TODO: Description
    """
    listchb = ["chb" + str(i) + "/" for i in range(11, 25)]
    listchb.append("../")

    return [item for item in folders_description if ~isin(item, listchb)]


def get_folders(folders_description):
    """
        TODO: Description
    """
    return [item for item in folders_description if item.find("/") != -1]


def get_files(folders_description):
    """
        TODO: Description
    """
    return [item for item in folders_description if item.find("/") == -1]


def download_chbmit(url_base, path_save):
    """

    Parameters
    ----------
    url_base :

    path_save :

    Returns
    -------


    """

    print("Downloading the folder information: {}".format(path_save))
    fold_save = Path(path_save)

    if not fold_save.exists():

        fold_save.mkdir(parents=True, exist_ok=True)

        # TODO: Se estiver usando python 3.6+, considere usar f-strings
        #  Senao, procure se habituar com o .format
        #  https://realpython.com/python-f-strings/
        # Deixar aqui até saber como fazer no format

        folders_description = download_item(
            url_base, path_save + "base.html", page=True)

        folders = get_folders(folders_description)
        description = get_files(folders_description)

        patient_url = zip_with_unique(url_base, folders)
        patient_item = zip_with_unique(path_save, folders)
        description_base = zip_with_unique(url_base, description)

        print("Downloading the folder files: {}".format(path_save))
        for item, name in zip(description_base, description):
            download_item(item, path_save + name, page=False)

        for item, name in zip(patient_url, patient_item):
            download_chbmit(item, name)
    else:
        print("Folder already exists\n Use load_dataset_chbmit")

    return patient_item


def load_dataset_boon(path_child_fold) -> [array]:
    """Function for reading the boon database, and return X and y.
    Also adapted from:
    https://mne-tools.github.io/mne-features/auto_examples/plot_seizure_example.html
    Parameters
    ----------

    path_child_fold : [str]
        List of strings with path to the dataset.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Data vectors, where n_samples is the number of samples
        and n_features is the number of features.
    y : array-like, shape (n_samples,)
        Target values.

    """

    data_segments = list()
    labels = list()

    for path in path_child_fold:

        f_names = [s for s in Path(path).iterdir() if str(
            s).lower().endswith(".txt")]

        for f_name in f_names:
            _data = read_csv(f_name, sep="\n", header=None)

            data_segments.append(_data.values.T[None, ...])

        if ("setE" in path) or ("setC" in path) or ("setD" in path):

            labels.append(ones((len(f_names),)))
        else:
            labels.append(zeros((len(f_names),)))

    X = concatenate(data_segments).squeeze()
    y = concatenate(labels, axis=0)

    return X, y


def preprocessing_split(X, y, test_size=.20, random_state=42) -> [array]:
    """Function to perform the train and test split
    and normalize the data set with Min-Max.

    Parameters
    ----------

    X : array-like, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.

    y : array-like, shape (n_samples,)
        Target values.

    test_size : float
        Value between 0 and 1 to indicate the
        percentage that will be used in the test.

    random_state : int
        Seed to be able to replicate split

    Returns
    -------
    X_train, X_test, y_train, y_test : list
        Separate data set for training and testing,
        standardized and correctly formatted for tensor.
    """
    # Separation of the data set in training and testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Min max scaler method.
    min_max = MinMaxScaler().fit(X_train)

    # Applying the re-scaling the training set and test set.
    X_train = min_max.transform(X_train)
    X_test = min_max.transform(X_test)

    # Removal of the last point present in the vector
    X_train = X_train[:, :4096]
    X_test = X_test[:, :4096]

    # Applying to reshape to match the input as tensor.
    X_train = reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test


def save_feature(df_train,
                 df_test,
                 value_encoding_dim,
                 path_dataset,
                 type_loss) -> [str]:
    """

    Parameters
    ----------
    df_train : DataFrame

    df_test : DataFrame

    value_encoding_dim : int,
        Size of the latent space that architecture will
        learn in the process of decoding and encoding.

    path_dataset : str,
        Path name where is the dataset in computer.

    type_loss : str
        Which loss function will be minimized in the learning proces,
        with the options: "mae" or "maae".

    Returns
    -------
        save_train_name : str
        Path name where the training dataset was saved on the computer.

        save_test_name : str
        Path name where the testing dataset was saved on the computer.
    """

    # Join pathname between a string that contains the base
    # pathname dataset and a folder called feature_learning,
    # which will be created to save the latent spaces
    # generated by AutoEnconder.
    path_save = join(path_dataset, "feature_learning")

    # Conversion of the pathname string to the class PurePath,
    # To use the class to create a folder on the system if it
    # doesn"t exist.
    fold_save = Path(path_save)

    # Formatted string to save loss type information and size dimensions
    name_train = "train_{}_{}.parquet".format(value_encoding_dim,
                                              type_loss)
    name_test = "test_{}_{}.parquet".format(value_encoding_dim,
                                            type_loss)

    # Join to take the path that we will save the train and test
    save_train_name = join(path_save, name_train)
    save_test_name = join(path_save, name_test)

    # If the folder does not exist, then create
    if not fold_save.exists():
        fold_save.mkdir(parents=True, exist_ok=True)

    # Saving as parquet to preserve the type.
    df_train.to_parquet(save_train_name, engine="pyarrow")
    df_test.to_parquet(save_test_name, engine="pyarrow")

    return save_train_name, save_test_name


def build_feature(X_train,
                  X_test,
                  y_train,
                  y_test,
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

    X_test : array-like  (n_samples, n_features)
        The input data to use in train process.

    Y_train : array-like  (n_samples, )
        The label data related to the X_train.

    Y_test : array-like  (n_samples, )
        The label data related to the X_test.

    The rest of the parameters and explanation of the
    variables are homonymous with those of the original class.

    Returns
    -------
    auto_encoder : AutoEnconder class object
        Object of the class already trained.

    path_train : str
        Path where the train set was saved.

    path_test  : str
        Path where the test set was saved.

    """

    print("Convert and save with value enconding dimension: {}".format(
        value_encoding_dim))

    # Initializing the Auto-Encoder model
    auto_encoder = AutoEnconder(epochs=epochs, batch_size=batch_size,
                                value_encoding_dim=value_encoding_dim,
                                type_loss=type_loss,
                                name_dataset=path_dataset)
    # For validation, as described in the text, we use the test dataset.
    auto_encoder.fit(X_train, X_test)

    # Data transformation
    X_train_encoded = auto_encoder.transform(X_train)
    X_test_encoded = auto_encoder.transform(X_test)

    # Conversion of numpy.array to a DataFrame
    df_train = DataFrame(X_train_encoded)
    # On the DataFrame each column has a numeric value,
    # which is converted to a string
    df_train.columns = df_train.columns.astype(str)
    # Join with class label
    df_train["class"] = y_train

    # Conversion of numpy.array to a DataFrame
    df_test = DataFrame(X_test_encoded)
    # On the DataFrame each column has a numeric value,
    # which is converted to a string
    df_test.columns = df_test.columns.astype(str)
    # Join with class label
    df_test["class"] = y_test

    path_train, path_test = save_feature(df_train=df_train,
                                         df_test=df_test,
                                         value_encoding_dim=value_encoding_dim,
                                         path_dataset=path_dataset,
                                         type_loss=type_loss)

    return auto_encoder, path_train, path_test
