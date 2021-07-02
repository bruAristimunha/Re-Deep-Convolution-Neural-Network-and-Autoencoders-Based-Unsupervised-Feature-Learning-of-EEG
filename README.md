# [Re] [Deep Convolution Neural Network and Autoencoders-Based Unsupervised Feature Learning of EEG Signals](https://doi.org/10.1109/ACCESS.2018.2833746)

<p align="center">
<a href="https://colab.research.google.com/github/bruAristimunha/Re-Deep-Convolution-Neural-Network-and-Autoencoders-Based-Unsupervised-Feature-Learning-of-EEG/blob/master/notebook/Jupyter_Paper_Re_Deep_Convolution_Neural_Network_and_Autoencoders_Based_Unsupervised_Feature_Learning_of_EEG_Signals.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

## Reproduction authors.

We have no affiliation with the original authors and our attempts to contact them have failed.

[Bruno Aristimunha](https://github.com/bruAristimunha)*<sup>1</sup>, [Diogo Eduardo Lima Alves](https://github.com/DiogoEduardo)*<sup>1</sup>, [Walter Hugo Lopez Pinaya](https://github.com/warvito) <sup>1,2</sup>, [Raphael Y. de Camargo](https://rycamargo.wixsite.com) <sup>1</sup>

> <sup>1</sup> Center for Mathematics, Computation and Cognition (CMCC), Federal Univesity of ABC (UFABC), Rua Arcturus, 03. Jardim Antares, São Bernardo do Campo, CEP 09606-070, SP, Brazil.

> <sup>2</sup> King’s College London, London, UK.

>*b.aristimunha@gmail.com, digmogle96@gmail.com

---

## Original paper authors.

Tingxi Wen <sup>2</sup>, Zhongnan Zhang* <sup>2</sup>

> <sup>2</sup> Software School, Xiamen University, Xiamen, China.

*zhongnan_zhang@xmu.edu.cn


### Abstract


This paper presents our efforts to reproduce and improve the results achieved by the authors of the original article. We follow the steps and models described in their article and the same public data sets of EEG Signals. Epilepsy affects more than 65 million people globally, and EEG Signals are critical to analyze and recognize epilepsy. Although the efforts in the last years, it is still challenging to extract useful information from these signals and select useful features in a diagnostic application. We construct a deep convolution network and autoencoders-based model (AE-CDNN) in order to perform unsupervised feature learning. We use the AE-CDNN to extract the features of the available data sets, and then we use some common classifiers to classify the features. The results obtained demonstrate that the proposed AE-CDNN outperforms the traditional feature extraction based classification techniques by achieving better accuracy of classification.


## Prerequisites for Reprodubility

Clone the repository and the branch:

```shell
git clone --recurse-submodules -j8 https://github.com/bruAristimunha/Re-Deep-Convolution-Neural-Network-and-Autoencoders-Based-Unsupervised-Feature-Learning-of-EEG.git ReScience-submission
```

Install Conda, we recommend the [tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Create a conda enviromnment as follows:

```shell
conda create --name eeg python=3.7 pip --yes
conda activate eeg
```
We recommend using a GPU, if you have it available. But if not, just ignore the first line below:

```shell
conda install -c anaconda tensorflow-gpu 
pip install -r ReScience-submission/requirements.txt
cd ReScience-submission/notebook
jupyter notebook
```

### Execute

Use Jupyter to open the notebook "Jupyter_Paper_Re_Deep_Convolution_Neural_Network_and_Autoencoders_Based_Unsupervised_Feature_Learning_of_EEG_Signals.ipynb" to reproduce the results