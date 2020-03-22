"""
TODO: Description about the file.


"""

from os import listdir
from os.path import join, isfile
from pathlib import Path
from zipfile import ZipFile
from re import findall

from glob import glob
from mne.io import read_raw_edf

from tqdm import tqdm_notebook

from warnings import filterwarnings

from bs4 import BeautifulSoup
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

from pandas import DataFrame
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from wget import download
from autoenconder import AutoEnconder
from sys import path

path.append("../../chb-mit/")

from patient import Patient


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


def download_item(url_base, name_base, page=True, range_ = (30, 50)):
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
        Parameter to be used to download the page"s html.

    Returns
    -------
    list : list [str]
        Filter list with files and folder, not included when id people > 10.

    """
    download(url_base, name_base)
    if page:
        base = open(name_base, "r").read()
        soup = BeautifulSoup(base, "html.parser")
        return filter_list([link.get("href") for link in soup.find_all("a")], range_ = range_)

    return None


def filter_list(folders_description, range_ = (11, 25)):
    """
        TODO: Description
    """
    listchb = ["chb" + str(i) + "/" for i in range(range_[0], range_[1])]
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
    Function to download the Chbmit dataset, [the 10 firsts patients only].
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
    # Or doing nothing :)
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

def save_feature_model(auto_encoder, 
                       path_dataset,
                       type_loss,
                       value_encoding_dim):
    
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
    enconder_name = "enconder_{}_{}.h5".format(type_loss, value_encoding_dim)
    enconder_name = join(path_save, enconder_name)

    auto_encoder.method_enconder.save(enconder_name)
    
    # Saving the auto enconder model        
    auto_enconder_name = "auto_enconder_{}_{}.h5".format(type_loss, value_encoding_dim)
    auto_enconder_name = join(path_save, auto_enconder_name)

    auto_encoder.method_enconder.save(auto_enconder_name)
    
        
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
.save('path_to_my_model.h5')

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
    
    save_feature_model(auto_encoder, path_dataset, type_loss, value_encoding_dim)
    
    return auto_encoder, path_train, path_test

def parallel_variance(count_a, avg_a, var_a, count_b, avg_b, var_b):
    """
    Function for calculating the variance in a 
    distributed way, adapted from:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    The modifications in the return form, 
    to allow the accumulation in a simpler way.
    

    """
    delta = avg_b - avg_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)

    # calculate variance
    var = (M2 / (count_a + count_b - 1))
    # calculate count
    count = count_a + count_b
    # calculate mean
    avg = (count_a*avg_a + count_b*avg_b) / (count)

    return count, avg, var



def get_variance_accumulated(path_dataset, range_ = (1,11)):
    """
    Calculation of accumulated variance in channels and files. 
    Parameter receives the path where the folder with files is located.
    Calculates the variance only in the first ten people.
    
    On the tested computer it took about 10 minutes going 
    through all the files and accumulating the variance.
    We filter warnings.
    """
    filterwarnings("ignore")
    
    accumulate_count = 0
    accumulate_avg = 0
    accumulate_var = 0
    
    selected_channels = ["time", "FP1-F7", "F7-T7", "T7-P7", 
                         "P7-O1", "FP1-F3", "F3-C3", "C3-P3", 
                         "P3-O1", "FP2-F4", "F4-C4", "C4-P4", 
                         "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", 
                         "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", 
                         "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"]
    
    for id_patient in tqdm_notebook(range(range_[0], range_[1]), desc="Patient"):
        
        path_files = join(path_dataset, "chb{0:0=2d}/*.edf".format(id_patient))

        files_in_folder = glob(path_files)
        for enum, file in enumerate(tqdm_notebook(files_in_folder, 
                                        desc="Files",
                                        leave=False)):
            
            variance_file = read_raw_edf(
                input_fname=file, verbose=0).to_data_frame(picks=["eeg"],
                                                          time_format="ms")
            
            # Removing channels that are not present in all files.
            variance_file = variance_file[variance_file.columns.intersection(selected_channels)]
            # Sorting the channels
            variance_file.sort_index(axis=1, inplace=True)

            if ((enum == 0) & (id_patient == 0)):
                accumulate_count = len(variance_file)
                accumulate_avg = variance_file.mean()
                accumulate_var = variance_file.var()

            else:
                accumulate_count, accumulate_avg, accumulate_var = parallel_variance(
                    accumulate_count, accumulate_avg, accumulate_var,
                    len(variance_file), variance_file.mean(), variance_file.var())

    return accumulate_count, accumulate_avg, accumulate_var


def get_variance_by_file(path_dataset, range_ = (1,11)):
    """
    TO-DO
    """
    
    filterwarnings("ignore")
    
    rank_variance = []

    
    selected_channels = ["time", "FP1-F7", "F7-T7", "T7-P7", 
                         "P7-O1", "FP1-F3", "F3-C3", "C3-P3", 
                         "P3-O1", "FP2-F4", "F4-C4", "C4-P4", 
                         "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", 
                         "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", 
                         "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"]
    
    for id_patient in tqdm_notebook(range(range_[0], range_[1]), desc="Patient"):
        
        path_files = join(path_dataset, "chb{0:0=2d}/*.edf".format(id_patient))

        files_in_folder = glob(path_files)
        for enum, file in enumerate(tqdm_notebook(files_in_folder, 
                                        desc="Files",
                                        leave=False)):
            
            variance_file = read_raw_edf(
                input_fname=file, verbose=0).to_data_frame(picks=["eeg"],
                                                          time_format="ms")
            
            # Removing channels that are not present in all files.
            variance_file = variance_file[variance_file.columns.intersection(selected_channels)]
            # Sorting the channels
            variance_file.sort_index(axis=1, inplace=True)
            
            variance_by_file = variance_file.var()

            rank_by_file = variance_by_file.sort_values()
            
            rank_variance.append(rank_by_file)
            
    return rank_variance

def get_variance_by_pearson(path_dataset, range_ = (1,11)):
    """
    TO-DO:
    """

    filterwarnings("ignore")

    var_pearson = []


    selected_channels = ["time", "FP1-F7", "F7-T7", "T7-P7", 
                         "P7-O1", "FP1-F3", "F3-C3", "C3-P3", 
                         "P3-O1", "FP2-F4", "F4-C4", "C4-P4", 
                         "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", 
                         "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", 
                         "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1"]

    for id_patient in tqdm_notebook(range(range_[0], range_[1]), desc="Patient"):

        accumulate_count = 0
        accumulate_avg = 0
        accumulate_var = 0

        path_files = join(path_dataset, "chb{0:0=2d}/*.edf".format(id_patient))

        files_in_folder = glob(path_files)
        for enum, file in enumerate(tqdm_notebook(files_in_folder, 
                                        desc="Files",
                                        leave=False)):

            variance_file = read_raw_edf(
                input_fname=file, verbose=0).to_data_frame(picks=["eeg"],
                                                          time_format="ms")

            # Removing channels that are not present in all files.
            variance_file = variance_file[variance_file.columns.intersection(selected_channels)]
            # Sorting the channels
            variance_file.sort_index(axis=1, inplace=True)

            if enum == 0:
                accumulate_count = len(variance_file)
                accumulate_avg = variance_file.mean()
                accumulate_var = variance_file.var()

            else:
                accumulate_count, accumulate_avg, accumulate_var = parallel_variance(
                    accumulate_count, accumulate_avg, accumulate_var,
                    len(variance_file), variance_file.mean(), variance_file.var())

        var_pearson.append(accumulate_var)

    return var_pearson

def filter_(n_array): 
    return filter(lambda x: x != [], n_array)

def split_4096(array):
    if len(array) >= 4096 and array != []:
        if len(array) % 4096 != 0:

            max_length = int((len(array)//4096)*4096)
            fix_size = array[:max_length]

        else:

            fix_size = array

        return vstack(array_split(fix_size, len(array)//4096))
    return []


def load_dataset_chbmit(path_save) -> [array]:
    """
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

    X_non = []
    X_seiz = []
    X = DataFrame()    
    
    for id_ in range(1, 11):
        print("Loading Patients nº {}".format(id_))
        pat = Patient(id_, path_save)

        non_epoch_array = list(map(split_4096, pat.get_non_seizures()))

        X_non.append(concatenate(non_epoch_array))

        s_clips = pat.get_seizure_clips()
        if s_clips != []:

            seiz_epoch = list(filter_(list(map(split_4096, s_clips))))

            X_seiz.append(concatenate(seiz_epoch))

    X_non = DataFrame(concatenate(X_non)).sample(n=200, random_state=42)

    X_seiz = DataFrame(concatenate(X_seiz)).sample(n=200, random_state=42)

    X = X_non.append(X_seiz)

    y = [0]*200+[1]*200
    
    return X.to_numpy(), y 