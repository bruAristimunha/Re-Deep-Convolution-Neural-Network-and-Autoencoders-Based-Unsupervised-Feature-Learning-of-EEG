"""
  TO-DO:  Description
"""
from os.path import join

from pandas import Series
from pandas import DataFrame
from matplotlib.pylab import subplots, ylabel, xlabel
from numpy import mean, around

from classification import read_feature_data

# TODO: Deixe aspas simples ou dupla consistente com o que eh usado no restante do codigo.
def plot_variance_accumulate(var):
    """
      TO-DO:  Description
    """
    fig, axes = subplots(figsize=(12, 5))

    axes = var.sort_values().drop('time').plot.bar(axes=axes)

    ylabel("Accumulated variance per channel")
    xlabel("EEG Channel")

    return fig


def plot_variance_by_file(variance_by_file):
    """
      TO-DO:  Description
    """
    fig, axes = subplots(figsize=(12, 5))

    var = [file.drop('time').sort_values().index[-1]
           for file in variance_by_file]

    axes = Series(var).value_counts().sort_values().plot.bar(axes=axes)

    ylabel("Accumulated rank per channel per file")
    xlabel("EEG Channel")

    return fig


def plot_variance_by_pearson(variance_per_person):
    """
      TO-DO:  Description
    """
    fig, axes = subplots(figsize=(12, 5))

    var = [file.drop('time').sort_values().index[-1]
           for file in variance_per_person]

    axes = Series(var).value_counts().sort_values().plot.bar(axes=axes)

    ylabel("Accumulated rank per channel per pearson")
    xlabel("EEG Channel")

    return fig


def original_experiments(x):
    """
      TO-DO:  Description
    """
    return x[(x['name_classifier'] != 'ensemble') & (x['Dimension'] != 256)]


def proposed_experiments(x):
    """
      TO-DO:  Description
    """
    return x[(x['name_classifier'] == 'ensemble') | (x['Dimension'] == 256)]


def table_classification_dimension(metrics, original=True):
    """
      TO-DO:  Description
    """
    if original:
        metrics = original_experiments(metrics)
    else:
        metrics = proposed_experiments(metrics)

    accuracy = metrics.groupby(
        ['Dimension', 'name_classifier'])['test_accuracy'].apply(mean).unstack()

    # Order with base in paper.
    accuracy = accuracy[['k_neighbors', 'svm_linear', 'svm_radial', 'decision_tree',
                         'random_forest', 'multi_layer', 'ada_boost', 'gaussian_nb']]

    return accuracy.apply(lambda x: around(x, decimals=4))


def table_classification_fold(metrics, original=True, dimension=2):
    """
      TO-DO:  Description
    """
    if original:
        metrics = original_experiments(metrics)
    else:
        metrics = proposed_experiments(metrics)

    accuracy = metrics[metrics['Dimension'] == dimension]
    # Order with base in paper.
    accuracy = accuracy.pivot_table(index='5-fold',
                                    columns='name_classifier',
                                    values='test_accuracy')

    accuracy = accuracy[['k_neighbors', 'svm_linear', 'svm_radial', 'decision_tree',
                         'random_forest', 'multi_layer', 'ada_boost', 'gaussian_nb']]

    return accuracy


def plot_average_accuracy(ae_d1_l1, ae_d1_l2, ae_d2_l1, ae_d2_l2):
    """
      TO-DO:  Description
    """
    ae_d1_l1 = original_experiments(ae_d1_l1).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    ae_d1_l1.name = 'AE-CDNN-L1'

    ae_d1_l2 = original_experiments(ae_d1_l2).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    ae_d1_l2.name = 'AE-CDNN-L2'

    ae_d2_l1 = original_experiments(ae_d2_l1).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    ae_d2_l1.name = 'AE-CDNN-L1'

    ae_d2_l2 = original_experiments(ae_d2_l2).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    ae_d2_l2.name = 'AE-CDNN-L2'

    fig, axes = subplots(nrows=1, ncols=2, figsize=(14, 10))

    df_1 = DataFrame([ae_d1_l1, ae_d1_l2]).T
    df_1.index = df_1.index.astype(str)

    df_2 = DataFrame([ae_d2_l1, ae_d2_l2]).T
    df_2.index = df_2.index.astype(str)

    axes[0] = df_1.plot.line(axes=axes[0], ylim=(0.5, 1), style='.-')
    axes[0].set(ylabel="Average accuracy")

    axes[0].set_title("Dataset 1")
    axes[0].legend(loc='lower right')

    axes[1] = df_2.plot.line(axes=axes[1], ylim=(0.5, 1), style='.-')
    axes[1].set_title("Dataset 2")
    axes[1].set(ylabel="Average accuracy")
    axes[1].legend(loc='lower right')

    return fig


def plot_average_accuracy_baseline(ae_d1_l1, ae_d1_l2, pca_d1, srp_d1,
                                   ae_d2_l1, ae_d2_l2, pca_d2, srp_d2):
    """
    TO-DO:  Description
    """
    # Auto-Enconder
    ae_d1_l1 = original_experiments(ae_d1_l1).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    ae_d1_l1.name = 'AE-CDNN-L1'

    ae_d1_l2 = original_experiments(ae_d1_l2).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    ae_d1_l2.name = 'AE-CDNN-L2'

    ae_d2_l1 = original_experiments(ae_d2_l1).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    ae_d2_l1.name = 'AE-CDNN-L1'

    ae_d2_l2 = original_experiments(ae_d2_l2).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    ae_d2_l2.name = 'AE-CDNN-L2'

    # Baseline

    pca_d1 = original_experiments(pca_d1).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    pca_d1.name = 'PCA'

    pca_d2 = original_experiments(pca_d2).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    pca_d2.name = 'PCA'

    srp_d1 = original_experiments(srp_d1).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    srp_d1.name = 'SRP'

    srp_d2 = original_experiments(srp_d2).groupby(
        ['Dimension'])['test_accuracy'].apply(mean)
    srp_d2.name = 'SRP'

    fig, axes = subplots(nrows=1, ncols=2, figsize=(14, 10))

    df_1 = DataFrame([ae_d1_l1, ae_d1_l2, pca_d1, srp_d1]).T
    df_1.index = df_1.index.astype(str)

    df_2 = DataFrame([ae_d2_l1, ae_d2_l2, pca_d2, srp_d2]).T
    df_2.index = df_2.index.astype(str)

    axes[0] = df_1.plot.line(axes=axes[0], ylim=(0.5, 1), style='.-')
    axes[0].set(ylabel="Average accuracy")

    axes[0].set_title("Dataset 1")
    axes[0].legend(loc='lower right')

    axes[1] = df_2.plot.line(axes=axes[1], ylim=(0.5, 1), style='.-')
    axes[1].set_title("Dataset 2")
    axes[1].set(ylabel="Average accuracy")
    axes[1].legend(loc='lower right')

    return fig

def encoded_class(x):
    """
    TODO:  Description
    """
    return '$P$' if x == 1 else '$N$'

def clean_xlabel(x):
    """
    TODO:  Description
    """
    return x.set(xlabel="")


def plot_feature_distribution(path_save, n_dims=4):
    """
    TODO:  Description
    """
    fig, axes = subplots(nrows=2, ncols=n_dims, figsize=(20, 10))

    path_base = join(path_save, "reduced")

    path_read = join(path_base, "ae_{}".format("mae"))
    data_frame_mae, y_mae = read_feature_data(path_read, n_dims)
    data_frame_mae.columns = ['$f_{}$'.format(int(name_col)+1)
                              for name_col in data_frame_mae.columns.values]

    data_frame_mae['class'] = y_mae.apply(encoded_class)

    axes[0] = data_frame_mae.boxplot(by='class', axes=axes[0]).reshape(-1)

    path_read = join(path_base, "ae_{}".format("mae"))
    data_frame_maae, y_maae = read_feature_data(path_read, n_dims)
    data_frame_maae.columns = ['$f_{}$'.format(int(name_col)+1)
                               for name_col in data_frame_maae.columns.values]

    data_frame_maae['class'] = y_maae.apply(encoded_class)

    axes[1] = data_frame_mae.boxplot(by='class', axes=axes[1]).reshape(-1)

    _ = list(map(clean_xlabel, axes[0]))
    _ = list(map(clean_xlabel, axes[1]))
    _ = fig.suptitle('')

    _ = axes[0][0].set(ylabel="AE-CDNN-L1")
    _ = axes[1][0].set(ylabel="AE-CDNN-L2")

    return fig


def plot_change_loss(history_l1, history_l2):
    """
    TODO:  Description
    """
    fig, axes = subplots(figsize=(15, 5), ncols=2)

    axes[0].plot(history_l1.history["loss"])
    axes[0].plot(history_l1.history["val_loss"])
    axes[0].set_title("AE-CDNN-L1")
    axes[0].set(ylabel="Loss values", xlabel="Iteration")
    axes[0].legend(["loss", "val_loss"], loc="lower left")

    axes[1].plot(history_l2.history["loss"])
    axes[1].plot(history_l2.history["val_loss"])
    axes[1].set_title("AE-CDNN-L2")
    axes[1].set(ylabel="Loss values", xlabel="Iteration")
    axes[1].legend(["loss", "val_loss"], loc="lower left")

    return fig
