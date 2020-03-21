import matplotlib.pyplot as plt
from pandas import Series

def plot_variance_accumulate(var):

    fig, ax = plt.subplots(figsize=(12, 5))

    ax = var.sort_values().drop('time').plot.bar(ax=ax)

    plt.ylabel("Accumulated variance per channel")
    plt.xlabel("EEG Channel")
    
    return fig


def plot_variance_by_file(variance_by_file):

    fig, ax = plt.subplots(figsize=(12, 5))
    
    var = [file.drop('time').sort_values().index[-1] for file in variance_by_file]
    
    ax = Series(var).value_counts().sort_values().plot.bar(ax=ax)

    plt.ylabel("Accumulated rank per channel per file")
    plt.xlabel("EEG Channel")
    
    return fig

def plot_variance_by_pearson(variance_per_person):

    fig, ax = plt.subplots(figsize=(12, 5))
    
    var = [file.drop('time').sort_values().index[-1] for file in variance_per_person]
    
    ax = Series(var).value_counts().sort_values().plot.bar(ax=ax)

    plt.ylabel("Accumulated rank per channel per pearson")
    plt.xlabel("EEG Channel")
    
    return fig