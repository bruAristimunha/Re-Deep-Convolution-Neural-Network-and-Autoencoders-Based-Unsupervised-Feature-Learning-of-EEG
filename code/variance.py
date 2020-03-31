"""
TODO: Description about the file.


"""

from glob import glob
from os.path import join
from warnings import filterwarnings

from mne.io import read_raw_edf
from tqdm import tqdm_notebook


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
    mean_squared = m_a + m_b + delta ** 2 * count_a * count_b / (count_a + count_b)

    # calculate variance
    var = (mean_squared / (count_a + count_b - 1))
    # calculate count
    count = count_a + count_b
    # calculate mean
    avg = (count_a * avg_a + count_b * avg_b) / count

    return count, avg, var


def get_variance_accumulated(path_dataset, range_=(1, 11)):
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
            variance_file = variance_file[variance_file.columns.intersection(
                selected_channels)]
            # Sorting the channels
            variance_file.sort_index(axis=1, inplace=True)

            if (enum == 0) & (id_patient == 0):
                accumulate_count = len(variance_file)
                accumulate_avg = variance_file.mean()
                accumulate_var = variance_file.var()

            else:
                accumulate_count, accumulate_avg, accumulate_var = parallel_variance(
                    accumulate_count, accumulate_avg, accumulate_var,
                    len(variance_file), variance_file.mean(), variance_file.var())

    return accumulate_count, accumulate_avg, accumulate_var


def get_variance_by_file(path_dataset, range_=(1, 11)):
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
        for file in tqdm_notebook(files_in_folder, desc="Files", leave=False):
            variance_file = read_raw_edf(
                input_fname=file, verbose=0).to_data_frame(picks=["eeg"],
                                                           time_format="ms")

            # Removing channels that are not present in all files.
            variance_file = variance_file[variance_file.columns.intersection(
                selected_channels)]
            # Sorting the channels
            variance_file.sort_index(axis=1, inplace=True)

            variance_by_file = variance_file.var()

            rank_by_file = variance_by_file.sort_values()

            rank_variance.append(rank_by_file)

    return rank_variance


def get_variance_by_person(path_dataset, range_=(1, 11)):
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
            variance_file = variance_file[variance_file.columns.intersection(
                selected_channels)]
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
