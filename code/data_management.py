"""Copyright 2019, Bruno Aristimunha.

This file is part of paper [Re] Deep Convolution
Neural Network and Autoencoders-Based Unsupervised
Feature Learning of EEG Signals.

--------------------------------------------
Data manipulation, input-output of files.
"""
# IO imports

# Imports for manipulating, accessing, reading and writing files.
from sys import path
from importlib import import_module
from os import listdir
from os.path import join, isfile
from pathlib import Path
from re import findall
from zipfile import ZipFile

from bs4 import BeautifulSoup
# Import used for array manipulation.
from numpy import (
    zeros,
    ones,
    concatenate,
    array,
    reshape,
    isin,
    array_split,
    vstack,
)
from pandas import (
    DataFrame,
    read_csv,
    read_parquet,
)

# Imports for array manipulation to prepare for dimension reduction.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Import used to download the data.
from wget import download

path.insert("chb-mit/", 0)
# Import of the class used to read the CHBMIT dataset.
patient = import_module('patient')
# get method or function from patient
Patient = getattr(patient, 'Patient')


def zip_with_unique(base, list_suffix):
    """Auxiliary function to generate a paired list with a unique element.

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
    """Download dataset function.

    Adapted from
    mne-tools.github.io/mne-features/auto_examples/plot_seizure_example.html
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

        for url, path_ in list(zip(urls, path_child_fold)):
            file_directory = download(url, path_)

            with ZipFile(file_directory, "r") as zip_ref:
                zip_ref.extractall(path_)

    return path_child_fold


def download_item(url_base, name_base, page=True, range_=(30, 50)):
    """Download the files in an isolated way.

    Used when the file listing is different from a folder.

    Parameters
    ----------
    url_base : str,
        Url to indicate where to download the dataset.
    name_base : str
        Pathname to indicate where to download the dataset.
    page : bool
        Parameter to be used to download the page"s html.
    range_ : tuple(int, int)
        Interval with patient id that will be excluded in
        the download. By default, it indexes an invalid range,
        that is, it downloads everything.
    Returns
    -------
    list : list [str]
        Filter list with files and folder, not included when id people > 10.
    """
    download(url_base, name_base)
    if page:
        base = open(name_base, "r").read()
        soup = BeautifulSoup(base, "html.parser")
        return filter_list([link.get("href")
                            for link in soup.find_all("a")], range_=range_)

    return None


def filter_list(folders_description, range_=(11, 25)):
    """Filter folder list to download."""
    listchb = ["chb" + str(i) + "/" for i in range(range_[0], range_[1])]
    listchb.append("../")

    return [item for item in folders_description if ~isin(item, listchb)]


def get_folders(folders_description):
    """Get folder in a list."""
    return [item for item in folders_description if item.find("/") != -1]


def get_files(folders_description):
    """Get files in a list."""
    return [item for item in folders_description if item.find("/") == -1]


def download_chbmit(url_base, path_save):
    """Download the Chbmit dataset.

    The 10 firsts patients only.
    This function also creates the folder, and if already exists the folder,
    returns the list of files.

    Analog to the function "download_bonn", with a URL parameter of difference.

    Parameters
    ----------
    url_base : str
        url from physionet.
    path_save : str
        pathname where to save.

    Returns
    -------
    path_child_fold : str,
        List of strings with the path where
        the dataset files were downloaded.
    """
    print("Downloading the folder information: {}".format(path_save))
    fold_save = Path(path_save)

    if not fold_save.exists():

        fold_save.mkdir(parents=True, exist_ok=True)

        folders_description = download_item(
            url_base, "{}base.html".format(path_save), page=True)

        folders = get_folders(folders_description)
        description = get_files(folders_description)

        patient_url = zip_with_unique(url_base, folders)
        patient_item = zip_with_unique(path_save, folders)
        description_base = zip_with_unique(url_base, description)

        print("Downloading the folder files: {}".format(path_save))
        for item, name in zip(description_base, description):
            download_item(item, "{}{}".format(path_save, name), page=False)

        for item, name in zip(patient_url, patient_item):
            download_chbmit(item, name)
    else:

        print("Folder already exists\nUse load_dataset_chbmit")

        onlyfolder = [folder
                      for folder in listdir(path_save)
                      if not isfile(join(url_base, folder))]

        patient_item = [folder
                        for folder in onlyfolder
                        if findall("chb([0-9])*", folder) != []]
    return patient_item


def load_dataset_boon(path_data: str) -> [array]:
    """Read the boon database, and return data and class.

    Also adapted from: https://mne-tools.github.io/mne-
    features/auto_examples/plot_seizure_example.html.

    Parameters
    ----------
    path_data : str
        PathName to the dataset.

    Returns
    -------
    data : array-like, shape (n_samples, n_features)
        Data vectors, where n_samples is the number of samples
        and n_features is the number of features.
    class_ : array-like, shape (n_samples,)
        Target values.
    """
    fold = Path(path_data)
    child_fold = ["setA", "setB", "setC", "setD", "setE"]

    path_child_fold = zip_with_unique(path_data, child_fold)

    if not fold.exists():
        print("Error file not find")
        return "Error"

    data_segments = list()
    labels = list()

    for path_ in path_child_fold:

        f_names = [s for s in Path(path_).iterdir() if str(
            s).lower().endswith(".txt")]

        for f_name in f_names:
            _data = read_csv(f_name, sep="\n", header=None)

            data_segments.append(_data.values.T[None, ...])

        if ("setE" in path_) or ("setC" in path_) or ("setD" in path_):

            labels.append(ones((len(f_names),)))
        else:
            labels.append(zeros((len(f_names),)))

    data = concatenate(data_segments).squeeze()
    class_ = concatenate(labels, axis=0)

    return data, class_


def filter_empty(n_array):
    """Filter empty values in a list."""
    return filter(lambda x: x != [], n_array)


def split_4096(n_array):
    """Split an array into n-arrays with size 4096 points each.

    Parameters
    ----------
    n_array : array-like

    Returns
    -------
    [array] : [array-like]
    """
    if len(n_array) >= 4096 and n_array != []:
        if len(n_array) % 4096 != 0:

            max_length = int((len(n_array) // 4096) * 4096)
            fix_size = n_array[:max_length]

        else:
            fix_size = n_array

        return vstack(array_split(fix_size, len(n_array) // 4096))
    return []


def check_exist_chbmit(path_save: str):
    """Check if the fold dataset exists."""
    fold = Path(path_save) / "as_dataset"

    if not fold.exists():
        fold.mkdir(parents=True, exist_ok=True)
        return False
    return True


def load_dataset_chbmit(path_save: str,
                        n_samples=200,
                        random_state=42) -> [array]:
    """Read the chbmit database, and return data and class.

    Split the dataset to the appropriate size.

    Parameters
    ----------
    random_state : int
    n_samples: int
    path_save: str

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Data vectors, where n_samples is the number of samples
        and n_features is the number of features.
    y : array-like, shape (n_samples,)
        Target values.
    """
    data_frame_non = []
    data_frame_seiz = []

    path_dataset = join(path_save, "as_dataset")
    name_dataset_non = join(path_dataset, "data_frame_non.parquet")
    name_dataset_seiz = join(path_dataset, "data_frame_seiz.parquet")

    if not check_exist_chbmit(path_save):

        print("Loading the files to create dataset")

        for person_id in range(1, 11):
            print("Loading Patients nº {}".format(person_id))
            pat = Patient(person_id, path_save)

            non_epoch_array = list(map(split_4096, pat.get_non_seizures()))

            data_frame_non.append(concatenate(non_epoch_array))

            s_clips = pat.get_seizure_clips()

            if s_clips != []:
                seiz_epoch = list(filter_empty(list(map(split_4096, s_clips))))

                data_frame_seiz.append(concatenate(seiz_epoch))

        data_frame_non = DataFrame(concatenate(data_frame_non))
        data_frame_non['class'] = [0] * len(data_frame_non)
        data_frame_non.columns = data_frame_non.columns.astype(str)
        data_frame_non.to_parquet(name_dataset_non, engine="pyarrow")

        data_frame_seiz = DataFrame(concatenate(data_frame_seiz))
        data_frame_seiz['class'] = [1] * len(data_frame_seiz)
        data_frame_seiz.columns = data_frame_seiz.columns.astype(str)
        data_frame_seiz.to_parquet(name_dataset_seiz, engine="pyarrow")

    else:
        print("Reading as dataframe")
        data_frame_non = read_parquet(name_dataset_non, engine="pyarrow")
        data_frame_seiz = read_parquet(name_dataset_seiz, engine="pyarrow")

    sample_non = data_frame_non.sample(n=n_samples, random_state=random_state)
    sample_seiz = data_frame_seiz.sample(
        n=n_samples, random_state=random_state)

    data_frame = sample_non.append(sample_seiz)

    return data_frame.drop('class', 1).to_numpy(), data_frame['class'].values


def preprocessing_split(data, class_,
                        test_size=.20, random_state=42) -> [array]:
    """Split the train and test split.

    Also normalize the data set with Min-Max.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.
    class_ : array-like, shape (n_samples,)
        Target values.
    test_size : float
        Value between 0 and 1 to indicate the
        percentage that will be used in the test.
    random_state : int
        Seed to be able to replicate split

    Returns
    -------
    data_train, data_test, class_train, class_test : list array
        Separate data set for training and testing,
        standardized and correctly formatted for tensor.
    """
    # Separation of the data set in training and testing.
    data_train, data_test, class_train, class_test = train_test_split(
        data, class_, test_size=test_size, random_state=random_state)

    # Min max scaler method.
    min_max = MinMaxScaler().fit(data_train)

    # Applying the re-scaling the training set and test set.
    data_train = min_max.transform(data_train)
    data_test = min_max.transform(data_test)

    # Removal of the last point present in the vector
    # Or doing nothing :)
    data_train = data_train[:, :4096]
    data_test = data_test[:, :4096]

    # Applying to reshape to match the input as tensor.
    data_train = reshape(data_train,
                         (data_train.shape[0], data_train.shape[1], 1))
    data_test = reshape(data_test,
                        (data_test.shape[0], data_test.shape[1], 1))

    return data_train, data_test, class_train, class_test


def read_feature_data(base_fold, dim):
    """Read feature dataset.

    Function to read the dataset already built by the auto-enconder.

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
    data : array
    class_ : array
    """
    name_reduced = join(base_fold, "reduced_dataset_{}.parquet".format(dim))
    data = read_parquet(name_reduced, engine="pyarrow").drop(["class"], 1)
    class_ = read_parquet(name_reduced, engine="pyarrow")["class"]

    return data, class_


def save_reduce(data_reduced,
                value_encoding_dim,
                path_dataset,
                name_type) -> [str]:
    """Save reduce dataset.

    Parameters
    ----------
    data_reduced : DataFrame

    value_encoding_dim : int,
        Size of the latent space that architecture will
        learn in the process of decoding and encoding.

    path_dataset : str,
        Path name where is the dataset in computer.

    name_type : str
        If feature learning:
            Which loss function will be minimized in the learning proces,
            with the options: "mae" or "maae".
        Else:
            Baseline method, with the options: "pca" or "srp"
    Returns
    -------
    save_reduced_name : str
        Path name where the reducing dataset was saved on the computer.
    """
    # Join pathname between a string that contains the base
    # pathname dataset and a folder called reduced,
    # which will be created to save the latent spaces
    # generated by AutoEnconder or Baseline.
    path_base = join(path_dataset, "reduced")

    # Conversion of the pathname string to the class PurePath,
    # To use the class to create a folder on the system if it
    # doesnot exist.
    fold_base = Path(path_base)

    # If the folder does not exist, then create
    if not fold_base.exists():
        fold_base.mkdir(parents=True, exist_ok=True)

    # Inside the folder to store the reduced space,
    # we save if it was a loss or baseline methode
    if name_type in ("mae", "maae"):
        path_save = join(path_base, "ae_{}".format(name_type))
    else:
        path_save = join(path_base, name_type)

    fold_save = Path(path_save)

    # If the folder does not exist, then create
    if not fold_save.exists():
        fold_save.mkdir(parents=True, exist_ok=True)

    # Formatted string to save size dimensions in name
    name_reduced = "reduced_dataset_{}.parquet".format(value_encoding_dim)

    # Join to take the path that we will save the train and test
    save_reduced_name = join(path_save, name_reduced)

    # Saving as parquet to preserve the type.
    data_reduced.to_parquet(save_reduced_name, engine="pyarrow")

    return save_reduced_name


def save_feature_model(auto_encoder,
                       path_dataset,
                       type_loss,
                       value_encoding_dim):
    """Save feature model.

    Parameters
    ----------
    auto_encoder : AutoEnconder,

    value_encoding_dim : int,
        Size of the latent space that architecture will
        learn in the process of decoding and encoding.

    path_dataset : str,
        Path name where is the dataset in computer.

    name_type : str
        If feature learning:
            Which loss function will be minimized in the learning proces,
            with the options: "mae" or "maae".
        Else:
            Baseline method, with the options: "pca" or "srp"
    """
    # Join pathname between a string that contains the base
    # pathname dataset and a folder called save_model,
    # which will be created to save the whole-model
    # Enconder and AutoEnconder.
    path_save = join(path_dataset, "save_model")

    # Conversion of the pathname string to the class PurePath,
    # To use the class to create a folder on the system if it
    # doesn"t exist.
    fold_save = Path(path_save)

    # If the folder does not exist, then create
    if not fold_save.exists():
        fold_save.mkdir(parents=True, exist_ok=True)

    # Saving the enconder model
    enconder_name = "enconder_{}_{}.h5".format(type_loss,
                                               value_encoding_dim)
    enconder_name = join(path_save, enconder_name)

    auto_encoder.method_enconder.save(enconder_name)

    # Saving the auto enconder model
    auto_enconder_name = "auto_enconder_{}_{}.h5".format(
        type_loss, value_encoding_dim)
    auto_enconder_name = join(path_save, auto_enconder_name)

    auto_encoder.method_enconder.save(auto_enconder_name)


def save_history_model(auto_encoder,
                       path_dataset,
                       type_loss,
                       value_encoding_dim):
    """Save history model.

    Parameters
    ----------
    auto_encoder : AutoEnconder,

    value_encoding_dim : int,
        Size of the latent space that architecture will
        learn in the process of decoding and encoding.

    path_dataset : str,
        Path name where is the dataset in computer.

    name_type : str
        with the options: "mae" or "maae".
    """
    # Join pathname between a string that contains the base
    # pathname dataset and a folder called save_model,
    # which will be created to save the whole-model
    # Enconder and AutoEnconder.
    path_save = join(path_dataset, "save_model")

    # Conversion of the pathname string to the class PurePath,
    # To use the class to create a folder on the system if it
    # doesn"t exist.
    fold_save = Path(path_save)

    # If the folder does not exist, then create
    if not fold_save.exists():
        fold_save.mkdir(parents=True, exist_ok=True)

    history = DataFrame(auto_encoder.method_autoenconder.history.history)
    history = history.reset_index()
    history['index'] = history['index']+1
    history = history.rename(columns={'index': 'epoch'})

    # Saving the auto enconder model
    history_name = "loss_history_{}_{}.parquet".format(
        type_loss, value_encoding_dim)
    history_name = join(path_save, history_name)

    history.to_parquet(history_name, engine="pyarrow")


def read_history_model(path_dataset, type_loss, value_encoding_dim):
    """Read history model.

    Parameters
    ----------
    value_encoding_dim : int,
        Size of the latent space that architecture will
        learn in the process of decoding and encoding.

    path_dataset : str,
        Path name where is the dataset in computer.

    name_type : str
        with the options: "mae" or "maae".
    """
    # Join pathname between a string that contains the base
    # pathname dataset and a folder called save_model,
    # which will be created to save the whole-model
    # Enconder and AutoEnconder.
    path_save = join(path_dataset, "save_model")

    # Saving the auto enconder model
    history_name = "loss_history_{}_{}.parquet".format(
        type_loss, value_encoding_dim)
    history_name = join(path_save, history_name)

    history = read_parquet(history_name, engine="pyarrow")
    return history


def save_classification(scores,
                        base_fold,
                        name_type,
                        cross_val):
    """Save classification results.

    Parameters
    ----------
    scores : DataFrame
        Results classsification aggregate.
    base_fold : str,
        Pathname where is the dataset in computer.
    name_type : str
        with the options: "mae" or "maae".
    cross_val : int
        value with cross values.
    """
    fold = Path(base_fold) / "save"

    if not fold.exists():
        fold.mkdir(parents=True, exist_ok=True)

    # Formatted string to save size dimensions in name
    name_classification = "classification_{}_cv{}.parquet".format(name_type,
                                                                  cross_val)
    # Join to take the path that we will save the train and test
    save_classification_name = join(fold, name_classification)

    # Saving as parquet to preserve the type.
    scores.to_parquet(save_classification_name, engine="pyarrow")


def get_original_results(id_table: str, path_original: str):
    """Read function to get the original results from article."""
    name_file = "Original_Tables - Table_{}.csv".format(id_table)

    return read_csv(join(path_original, name_file), index_col="Dimension")
