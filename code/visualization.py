import matplotlib.pyplot as plt

def checking_variance(var):

    fig, ax = plt.subplots(figsize=(12, 5))

    ax = var.sort_values().drop('time').plot.bar(ax=ax)

    plt.ylabel("Accumulated variance per channel")
    plt.xlabel("EEG Channel")
    
    return fig