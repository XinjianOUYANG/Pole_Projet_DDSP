# Pole_Projet_DDSP(This is an ongoing project)
 
### Introduction of DDSP and our project

* Reprinted article: [DDSP paper and codes summary](https://www.cmwonderland.com/blog/2020/03/01/ddsp_sum/)

* [Presentation of DDSP](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/620ced17411f5a450748d51fc227040f787b98c1/T1/DDSP_T1.pdf)

* [Official DDSP library](https://github.com/magenta/ddsp)

* [An unofficial one](https://github.com/Manza12/DDSP) but valuable

* Description of our project [Projet 2A - Simon Leglaive.pdf](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/76cc7467678985e2750b62647fb39c616d7223e7/PDF_documents/Projet%202A%20-%20Simon%20Leglaive.pdf)

### The main problems we have:
* Computational power.
* Loading the model trained with z-encoder.


### Installation

Requires python = 3.8 and tensorflow version >= 2.1.0, but the core library runs in either eager or graph mode.

It would be better to create a new environment in Anaconda.

* pip install --upgrade pip

* pip install --upgrade ddsp

### Overview

* [utils.py](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/7568f3114fca4b9c59558036f09a01a27b756d39/utils.py): Some useful functions such as plotting spectrogram, adding noise etc.
 
 * [train_autoencoder.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/7568f3114fca4b9c59558036f09a01a27b756d39/train_autoencoder.ipynb): Training a DDSP model in Google Colab.
 
 * [showing_models_colab](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/7568f3114fca4b9c59558036f09a01a27b756d39/showing_models_colab.ipynb): Loading a trained DDSP model in Google Colab.(It may take  much time since we have to upload the model and audio files)

* [showing_models_local](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/7568f3114fca4b9c59558036f09a01a27b756d39/showing_models_local.ipynb): Loading a trained DDSP model in local environment.(Recommended, and we have to set the paths of model file and audio file manually)

* [timbre_transform.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/c3b213e64ba9fbbdf3cf16d0467e2896124bbdb7/timbre_transfer.ipynb): Using pretrained DDSP models to transform the timbre of an audio in Google Colab
