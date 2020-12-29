# VIGNet
Tensorflow implementation of [VIGNet: A Deep Convolutional Neural Network for EEG-based Driver Vigilance Estimation](https://ieeexplore.ieee.org/abstract/document/9061668).

### Requirements
tensorflow (1.14.0)\
tensorboard (2.2.2)\
tqdm (4.48.0)\
matplotlib (3.3.0)\
numpy (1.19.0)\
scikit-learn (0.23.2)


### Datasets
We used publicly available [SEED-VIG dataset](https://iopscience.iop.org/article/10.1088/1741-2552/aa5a98/meta?casa_token=zMmqflOHEYYAAAAA:F7YusFzBVULbjWBmoy39cvGI9RPMrUrDIOF_s1azdKrH1L0KJW9Cw_NuqFspM5OsRjMpECCpwtne)
>- 23 trials, ~ two hours EEG signal/trial
>- 17 electrode channels, 200Hz sampling rate
>- This dataset is labeled by [PERCLOS level](https://iopscience.iop.org/article/10.1088/1741-2552/aa5a98/meta?casa_token=zMmqflOHEYYAAAAA:F7YusFzBVULbjWBmoy39cvGI9RPMrUrDIOF_s1azdKrH1L0KJW9Cw_NuqFspM5OsRjMpECCpwtne)


### How to run
Mode:\
#0 Pre-training a classifier\
#1 Training the counterfactual map generator

1. Pre-training a classifier
>- `training.py --mode=0`

2. Training the counterfactual map generator
>- Set the classifier and encoder weight for training (freeze)
>- Change the mode from 0 to 1 on Config.py
  >- `training.py --mode=1`


### Config.py of each dataset 
data_path = Raw dataset path\
save_path = Storage path to save results such as tensorboard event files, model weights, etc.\
cls_weight_path = Pre-trained classifier weight path obtained in mode#0 setup\
enc_weight_path = Pre-trained encoder weight path obtained in mode#0 setup
