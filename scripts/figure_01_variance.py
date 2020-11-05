base_fold = ".."

path_original = "{}/data/original_results/".format(base_fold)
path_figure = "{}/article/figure".format(base_fold)
path_table = "{}/article/table".format(base_fold)


import sys
import os
import click
import matplotlib.pylab as plt
from shutil import rmtree
from pathlib import Path

plt.style.use("seaborn")
plt.style.use("seaborn-poster")

sys.path.append("{}/code".format(base_fold))
sys.path.append("{}/code/chb-mit/".format(base_fold))

from data_management import (
    download_chbmit,
    get_original_results,
    load_dataset_boon,
    load_dataset_chbmit,
    preprocessing_split,
)

from variance import (
    get_variance_accumulated,
    get_variance_by_file,
    get_variance_by_person,)

from visualization import (
    plot_variance_accumulate,
    plot_variance_by_file,
    plot_variance_by_person,)

@click.command()
@click.option('--path', default='..', 
              help='Base path to where you are. If you are running inside the repository\'s script folder, the default will be \'..\'')
@click.option('--download', default=1, 
              help='If you want to run from scratch, default is 1')
@click.option('--path_figure', default=".", 
              help='Path to where you want to save the figure.')

def make_figure_01(path, download, path_figure):
    
    chbmit_url = "https://physionet.org/files/chbmit/1.0.0/"
    path_chbmit = "{}/data/chbmit/".format(path)
    
    if download == 1:
        if Path(path_chbmit).exists():
            print("Removing folder")
            rmtree(path_chbmit, ignore_errors=True)

        print("Downloading Files")
        chbmit_path_child_fold = download_chbmit(
                                    url_base=chbmit_url,
                                    path_save=path_chbmit)

    else:
        if not (Path(path_chbmit).exists()):

            print("No folder find you must run with the option download equals 1.")
            
            return
  
    print("Processing the first scenario")

    variance_by_file = get_variance_by_file(path_chbmit)
    
    fig_by_file = plot_variance_by_file(variance_by_file)

    print("Saving the first scenario in the path {}".format("{}/variance_per_file.pdf".format(path_figure)))

    plt.savefig("{}/variance_per_file.pdf".format(path_figure), 
                bbox_inches="tight", dpi=600)

    print("Processing the second scenario")

    variance_per_person = get_variance_by_person(path_chbmit, 
                                                 range_= (1,11))

    print("Saving the second scenario in the path {}".format("{}/variance_per_person.pdf".format(path_figure)))
    fig_by_person = plot_variance_by_person(variance_per_person)
    
    plt.savefig("{}/variance_per_person.pdf".format(path_figure),
                bbox_inches="tight", dpi=600)  
          
    print("Processing the third scenario")

    accumulate_var = get_variance_accumulated(path_chbmit, 
                                              range_= (1,11))

    print("Saving the third scenario in the path {}".format("{}/variance_all.pdf".format(path_figure)))

    fig_accumulate = plot_variance_accumulate(accumulate_var)
    plt.savefig("{}/variance_all.pdf".format(path_figure), 
                bbox_inches="tight", dpi=600)


if __name__ == '__main__':
    make_figure_01()




