# Pole_Projet_DDSP

### Introduction of DDSP and our project

* Reprinted article: [DDSP paper and codes summary](https://www.cmwonderland.com/blog/2020/03/01/ddsp_sum/)
* [Presentation of DDSP](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/620ced17411f5a450748d51fc227040f787b98c1/T1/DDSP_T1.pdf)
* [Official DDSP library](https://github.com/magenta/ddsp)
* [An unofficial DDSP library](https://github.com/Manza12/DDSP) but valuable
* Description of our project [Projet 2A - Simon Leglaive.pdf](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/76cc7467678985e2750b62647fb39c616d7223e7/PDF_documents/Projet%202A%20-%20Simon%20Leglaive.pdf)

* Gaussian Mixture Model [sklearn GMM](https://scikit-learn.org/stable/modules/mixture.html#gmm)

### Packages Installation

Requires python = 3.6~3.8 and tensorflow version >= 2.4.0, but the core library runs in either eager or graph mode.

It is recommended  to create a new environment in Anaconda.

    pip install --upgrade pip
    pip install --upgrade ddsp

Required packages:

    numpy, math, base64, io. os, ddsp, gin, pickle, matplotlib, scipy, librosa, soundfile...
    
### Overview

* [utils.py](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/7568f3114fca4b9c59558036f09a01a27b756d39/utils.py): Some useful functions such as plotting spectrogram, adding noise etc.
* [train_autoencoder_colab.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/7568f3114fca4b9c59558036f09a01a27b756d39/train_autoencoder.ipynb)(from library tutorials): Training a DDSP model in Google Colab.
* [train_local_ae.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/main/train_local_ae.ipynb): Training a DDSP model with z encoder locally.
* [train_local_solo.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/main/train_local_solo.ipynb): Training a DDSP model without z encoder locally.
* [showing_models_colab.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/7568f3114fca4b9c59558036f09a01a27b756d39/showing_models_colab.ipynb)(from library tutorials): Loading a trained DDSP model in Google Colab.(It may take  much time since we have to upload the model and audio files)
* [showing_models_local.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/7568f3114fca4b9c59558036f09a01a27b756d39/showing_models_local.ipynb): Loading a trained DDSP model in local environment.(Recommended, and we have to set the paths of model file and audio file manually)
* [timbre_transform_colab.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/c3b213e64ba9fbbdf3cf16d0467e2896124bbdb7/timbre_transfer.ipynb)(from library tutorials): Using pretrained DDSP models to transform the timbre of an audio in Google Colab
* [T3_z_generator.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/main/T3_z_generator.ipynb): Extracting z datasets from audios using the pretrained model.
* [T3_synthesizer.ipynb](https://github.com/XinjianOUYANG/Pole_Projet_DDSP/blob/main/T3_synthesizer.ipynb): A generative DDSP model using the GMM and predefined pitch and loudness profiles and synthesing an audio sound.
