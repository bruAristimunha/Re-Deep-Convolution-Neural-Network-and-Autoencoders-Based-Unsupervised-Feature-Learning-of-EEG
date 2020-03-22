
from pandas import Series
from pandas import DataFrame
from matplotlib.pylab import subplots, ylabel, xlabel
from classification import read_feature_data
from os.path import join


def plot_variance_accumulate(var):

    fig, ax = subplots(figsize=(12, 5))

    ax = var.sort_values().drop('time').plot.bar(ax=ax)

    ylabel("Accumulated variance per channel")
    xlabel("EEG Channel")
    
    return fig


def plot_variance_by_file(variance_by_file):

    fig, ax = subplots(figsize=(12, 5))
    
    var = [file.drop('time').sort_values().index[-1] for file in variance_by_file]
    
    ax = Series(var).value_counts().sort_values().plot.bar(ax=ax)

    ylabel("Accumulated rank per channel per file")
    xlabel("EEG Channel")
    
    return fig

def plot_variance_by_pearson(variance_per_person):

    fig, ax = subplots(figsize=(12, 5))
    
    var = [file.drop('time').sort_values().index[-1] for file in variance_per_person]
    
    ax = Series(var).value_counts().sort_values().plot.bar(ax=ax)

    ylabel("Accumulated rank per channel per pearson")
    xlabel("EEG Channel")
    
    return fig


def plot_average_accuracy(ae_d1_l1, ae_d1_l2, ae_d2_l1, ae_d2_l2):

    fig, ax = subplots(nrows=1, ncols=2, figsize=(14, 10))

    df_1 = DataFrame([ae_d1_l1, ae_d1_l2]).T
    df_1.index = df_1.index.astype(str)

    df_2 = DataFrame([ae_d2_l1, ae_d2_l2]).T
    df_2.index = df_2.index.astype(str)


    ax[0] = df_1.plot.line(ax=ax[0], ylim=(0.5, 1), style='.-')
    ax[0].set(ylabel="Average accuracy")

    ax[0].set_title("Dataset 1")
    ax[0].legend(loc='lower right')


    ax[1] = df_2.plot.line(ax=ax[1], ylim=(0.5, 1), style='.-')
    ax[1].set_title("Dataset 2")
    ax[1].set(ylabel="Average accuracy")
    ax[1].legend(loc='lower right')

    return fig



def encoded_class(x): return '$P$' if x == 1 else '$N$'

def clean_xlabel(x): return x.set(xlabel="")


def plot_feature_distribution(path_save, n_dims=4):

    fig, ax = subplots(nrows=2, ncols=n_dims, figsize=(20, 10))
    path_read = join(path_save, "feature_learning")

    X_mae, y_mae = read_feature_data(path_read, n_dims, 'mae')
    X_mae.columns = ['$f_{}$'.format(int(name_col)+1)
                     for name_col in X_mae.columns.values]

    X_mae['class'] = y_mae.apply(encoded_class)

    ax[0] = X_mae.boxplot(by='class', ax=ax[0]).reshape(-1)

    X_maae, y_maae = read_feature_data(path_read, n_dims, 'maae')
    X_maae.columns = ['$f_{}$'.format(int(name_col)+1)
                      for name_col in X_maae.columns.values]
    
    X_maae['class'] = y_maae.apply(encoded_class)

    ax[1] = X_mae.boxplot(by='class', ax=ax[1]).reshape(-1)

    _ = list(map(clean_xlabel, ax[0]))
    _ = list(map(clean_xlabel, ax[1]))
    _ = fig.suptitle('')

    _ = ax[0][0].set(ylabel="AE-CDNN-L1")
    _ = ax[1][0].set(ylabel="AE-CDNN-L2")

    return fig

def plot_change_loss(history_l1, history_l2):
    fig, ax = subplots(figsize=(15, 5), ncols=2)

    ax[0].plot(history_l1.history["loss"])
    ax[0].plot(history_l1.history["val_loss"])
    ax[0].set_title("AE-CDNN-L1")
    ax[0].set(ylabel = "Loss values",xlabel = "Iteration")
    ax[0].legend(["loss", "val_loss"], loc="lower left")

    ax[1].plot(history_l2.history["loss"])
    ax[1].plot(history_l2.history["val_loss"])
    ax[1].set_title("AE-CDNN-L2")
    ax[1].set(ylabel = "Loss values",xlabel = "Iteration")
    ax[1].legend(["loss", "val_loss"], loc="lower left")

    return fig