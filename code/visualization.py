"""Copyright 2019, Bruno Aristimunha.

This file is part of paper [Re] Deep Convolution
Neural Network and Autoencoders-Based Unsupervised
Feature Learning of EEG Signals.

--------------------------------------------
Plot Function and table function.
"""
from os.path import join

from pandas import (
    Series,
    DataFrame,
    melt,
)

from matplotlib.pylab import (
    subplots,
    ylabel,
    xlabel,
)
from numpy import mean

from sklearn.utils._testing import ignore_warnings

from seaborn import boxplot, lmplot

from classification import read_feature_data


def regression_plot(metrics, name_metric="accuracy"):
    """Regression plot with dimension values, by classifiers."""
    reprod_table = table_classification_dimension(
        metrics, False, False, name_metric)

    graph = melt(reprod_table.reset_index(), id_vars=["Dimension"])
    graph.columns = ["Number of Feature - m",
                     "Classifier",
                     name_metric]

    g_plot = lmplot(x="Number of Feature - m", y=name_metric,
                    hue="Classifier", col="Classifier",
                    data=graph, aspect=1, col_wrap=3,
                    x_jitter=.1, sharex=True,
                    line_kws={"lw": 2, "ls": "--"},
                    seed=42)

    g_plot = g_plot.set_axis_labels("", name_metric)

    g_plot = g_plot.set(xlim=(-30, 280), ylim=(0.5, 1),
                        xticks=[0, 32, 64, 128, 256]).fig.subplots_adjust(wspace=.4)

    return g_plot


def plot_variance_accumulate(var):
    """Variance accumulate plot, by Channel."""
    fig, axes = subplots(figsize=(12, 5))

    axes = var.sort_values().drop("time").plot.bar(ax=axes)

    ylabel("Accumulated variance per channel")
    xlabel("EEG Channel")

    return fig


def plot_variance_by_file(variance_by_file):
    """Variance accumulate plot by file."""
    fig, axes = subplots(figsize=(12, 5))

    var = [file.drop("time").sort_values().index[-1]
           for file in variance_by_file]

    axes = Series(var).value_counts().sort_values().plot.bar(ax=axes)

    ylabel("Accumulated rank per channel per file")
    xlabel("EEG Channel")

    return fig


def plot_variance_by_person(variance_per_person):
    """Variance accumulate plot by person."""
    fig, axes = subplots(figsize=(12, 5))

    var = [file.drop("time").sort_values().index[-1]
           for file in variance_per_person]

    axes = Series(var).value_counts().sort_values().plot.bar(ax=axes)

    ylabel("Accumulated rank per channel per person")
    xlabel("EEG Channel")

    return fig


def original_experiments(data_fram):
    """Filter just original experiments."""
    return data_fram[(data_fram["name_classifier"] != "ensemble") &
                     (data_fram["Dimension"] != 256)]


def proposed_experiments(data_fram):
    """Filter just the proposed experiments."""
    return data_fram[(data_fram["name_classifier"] == "ensemble") |
                     (data_fram["Dimension"] == 256)]


def table_classification_dimension(metrics, original=True,
                                   proposed=True, metric="accuracy"):
    """Select one metric from the table metrics by dimension."""
    if original:
        metrics = original_experiments(metrics)
        order_col = ["k_neighbors", "svm_linear", "svm_radial",
                     "decision_tree", "random_forest", "multi_layer",
                     "ada_boost", "gaussian_nb"]
    elif proposed:
        metrics = proposed_experiments(metrics)
    else:
        order_col = ["k_neighbors", "svm_linear", "svm_radial",
                     "decision_tree", "random_forest", "multi_layer",
                     "ada_boost", "gaussian_nb", "ensemble"]

    values_metrics = metrics.groupby(
        ["Dimension", "name_classifier"])["test_{}".format(metric)].apply(mean).unstack()

    # Order with base in paper.
    values_metrics = values_metrics[order_col]

    values_metrics["average"] = values_metrics.mean(axis=1)

    return values_metrics


def table_classification_fold(metrics, original=True,
                              proposed=True, dimension=2,
                              metric="accuracy"):
    """Select one dimension and one metric from the metrics table by fold."""
    if original:
        metrics = original_experiments(metrics)
    elif proposed:
        metrics = proposed_experiments(metrics)
    else:
        pass

    values_metrics = metrics[metrics["Dimension"] == dimension]
    # Order with base in paper.
    values_metrics = values_metrics.pivot_table(index="5-fold",
                                                columns="name_classifier",
                                                values="test_{}".format(metric))

    values_metrics = values_metrics[["k_neighbors", "svm_linear", "svm_radial", "decision_tree",
                                     "random_forest", "multi_layer", "ada_boost", "gaussian_nb"]]

    return values_metrics


def plot_average_metric(ae_d1_l1, ae_d1_l2, ae_d2_l1, ae_d2_l2,
                        names, metric="accuracy"):
    """Plot the Average result by dimension, and loss function."""
    ae_d1_l1 = original_experiments(ae_d1_l1).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    ae_d1_l1.name = names[0]

    ae_d1_l2 = original_experiments(ae_d1_l2).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    ae_d1_l2.name = names[1]

    ae_d2_l1 = original_experiments(ae_d2_l1).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    ae_d2_l1.name = names[0]

    ae_d2_l2 = original_experiments(ae_d2_l2).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    ae_d2_l2.name = names[1]

    fig, axes = subplots(nrows=1, ncols=2, figsize=(14, 7))

    df_1 = DataFrame([ae_d1_l1, ae_d1_l2]).T
    df_1.index = df_1.index.astype(str)

    df_2 = DataFrame([ae_d2_l1, ae_d2_l2]).T
    df_2.index = df_2.index.astype(str)

    axes[0] = df_1.plot.line(ax=axes[0], ylim=(0.4, 1), style=".-")
    axes[0].set(ylabel="Average {}".format(metric))

    axes[0].set_title("Dataset 1")
    axes[0].legend(loc="lower right")

    axes[1] = df_2.plot.line(ax=axes[1], ylim=(0.4, 1), style=".-")
    axes[1].set_title("Dataset 2")
    axes[1].set(ylabel="Average {}".format(metric))
    axes[1].legend(loc="lower right")

    return fig


def plot_average_metric_baseline(ae_d1_l1, ae_d1_l2, pca_d1, srp_d1,
                                 ae_d2_l1, ae_d2_l2, pca_d2, srp_d2,
                                 metric="accuracy",
                                 name=None):
    """Plot the Average result by dimension and loss function, and baseline."""
    # Auto-Enconder
    ae_d1_l1 = original_experiments(ae_d1_l1).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    ae_d1_l1.name = name[0]

    ae_d1_l2 = original_experiments(ae_d1_l2).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    ae_d1_l2.name = name[1]

    ae_d2_l1 = original_experiments(ae_d2_l1).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    ae_d2_l1.name = name[0]

    ae_d2_l2 = original_experiments(ae_d2_l2).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    ae_d2_l2.name = name[1]

    # Baseline

    pca_d1 = original_experiments(pca_d1).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    pca_d1.name = name[2]

    pca_d2 = original_experiments(pca_d2).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    pca_d2.name = name[2]

    srp_d1 = original_experiments(srp_d1).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    srp_d1.name = name[3]

    srp_d2 = original_experiments(srp_d2).groupby(
        ["Dimension"])["test_{}".format(metric)].apply(mean)
    srp_d2.name = name[3]

    fig, axes = subplots(nrows=1, ncols=2, figsize=(14, 7))

    df_1 = DataFrame([ae_d1_l1, ae_d1_l2, pca_d1, srp_d1]).T
    df_1.index = df_1.index.astype(str)

    df_2 = DataFrame([ae_d2_l1, ae_d2_l2, pca_d2, srp_d2]).T
    df_2.index = df_2.index.astype(str)

    axes[0] = df_1.plot.line(ax=axes[0], ylim=(0.4, 1), style=".-")
    axes[0].set(ylabel="Average accuracy")

    axes[0].set_title("Dataset 1")
    axes[0].legend(loc="lower right")

    axes[1] = df_2.plot.line(ax=axes[1], ylim=(0.4, 1), style=".-")
    axes[1].set_title("Dataset 2")
    axes[1].set(ylabel="Average accuracy")
    axes[1].legend(loc="lower right")

    return fig


def encoded_class(data):
    """Enconder the class for boxplot."""
    return "$P$" if data == 1 else "$N$"


def clean_xlabel(axes):
    """Clean the xlabel."""
    return axes.set(xlabel="")


@ignore_warnings(category=UserWarning)
def plot_feature_distribution(path_save, n_dims=4,
                              names=None):
    """Plot the feature distribution in the reduced dataset."""
    fig, axes = subplots(nrows=2, ncols=n_dims,
                         figsize=(20, 10))

    path_base = join(path_save, "reduced")

    path_read = join(path_base, "ae_{}".format("mae"))
    data_frame_mae, y_mae = read_feature_data(path_read, n_dims)
    data_frame_mae.columns = ["$f_{}$".format(int(name_col)+1)
                              for name_col in data_frame_mae.columns.values]

    data_frame_mae["class"] = y_mae.apply(encoded_class)

    axes[0] = data_frame_mae.boxplot(by="class", ax=axes[0]).reshape(-1)

    path_read = join(path_base, "ae_{}".format("maae"))
    data_frame_maae, y_maae = read_feature_data(path_read, n_dims)
    data_frame_maae.columns = ["$f_{}$".format(int(name_col)+1)
                               for name_col in data_frame_maae.columns.values]

    data_frame_maae["class"] = y_maae.apply(encoded_class)

    axes[1] = data_frame_maae.boxplot(by="class", ax=axes[1]).reshape(-1)

    _ = list(map(clean_xlabel, axes[0]))
    _ = list(map(clean_xlabel, axes[1]))
    _ = fig.suptitle("")

    _ = axes[0][0].set(ylabel=names[0])
    _ = axes[1][0].set(ylabel=names[1])

    return fig


def plot_change_loss(history_l1, history_l2,
                     names=None):
    """Plot the loss variance in auto-enconder."""
    fig, axes = subplots(figsize=(15, 5), ncols=2)

    axes[0].plot(history_l1["loss"])
    axes[0].plot(history_l1["val_loss"])
    axes[0].set_title(names[0])
    axes[0].set(ylabel="Loss values", xlabel="Iteration")
    axes[0].legend(["loss", "val_loss"], loc="best")

    axes[1].plot(history_l2["loss"])
    axes[1].plot(history_l2["val_loss"])
    axes[1].set_title(names[1])
    axes[1].set(ylabel="Loss values", xlabel="Iteration")
    axes[1].legend(["loss", "val_loss"], loc="best")

    return fig


def boxplot_difference(reprod_table, origin_table):
    """Boxplot the difference beetween reproduced and original."""
    diff = melt(reprod_table-origin_table)
    diff.columns = ["", ""]

    fig, axes = subplots(figsize=(17, 5), nrows=2, sharex=True,
                         sharey=True)

    diff.columns = ["Classifier",
                    "Difference between\n obtained and reported accuracy."]

    axes[0] = boxplot(data=diff,
                      y="Difference between\n obtained and reported accuracy.",
                      x="Classifier", ax=axes[0])

    axes[1] = (reprod_table-origin_table).T.plot.bar(ax=axes[1])

    _ = axes[1].set(xlabel="Classifiers name")
    _ = axes[1].set(
        ylabel="Difference between\n obtained and reported accuracy.")
    _ = axes[0].set(ylabel="")

    bbox_anchor = (fig.subplotpars.left-0.5, fig.subplotpars.top-2.5,
                   fig.subplotpars.right-fig.subplotpars.left, .1)

    _ = axes[1].legend(bbox_to_anchor=bbox_anchor,
                       loc="lower right",
                       ncol=4, title="Dimension",
                       fancybox=True, shadow=True,
                       title_fontsize=16)

    return fig


def table_export_latex(path_save, dataset,
                       name_dataset, metric,
                       name_type, original,
                       proposed):
    """Export and save metrics as latex."""
    data = table_classification_dimension(dataset[name_type],
                                          original, proposed,
                                          metric=metric)

    metric_name = metric.capitalize()

    title = "{} values obtained by the same methodology - {} Dataset with {}.".format(metric_name,
                                                                                      name_dataset,
                                                                                      name_type)
    label_latex = "{}_{}_{}-reproduction".format(metric,
                                                 name_dataset,
                                                 name_type)

    title_name = "{}.tex".format(label_latex)

    columns = data.columns.to_list()

    data.to_latex(buf=join(path_save, title_name),
                  caption=title,
                  label=label_latex,
                  bold_rows=True,
                  columns=columns)

    return data, data.style.set_caption(title)
