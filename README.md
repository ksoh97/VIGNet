# VIGNet
Tensorflow implementation of [VIGNet: A Deep Convolutional Neural Network for EEG-based Driver Vigilance Estimation](https://ieeexplore.ieee.org/abstract/document/9061668).

### Requirements
tensorflow (1.14.0)\
tensorboard (2.2.0)\
matplotlib (3.3.0)\
numpy (1.19.0)\
scikit-learn (0.23.2)


### Datasets
We used publicly available [SEED-VIG dataset](https://iopscience.iop.org/article/10.1088/1741-2552/aa5a98/meta?casa_token=zMmqflOHEYYAAAAA:F7YusFzBVULbjWBmoy39cvGI9RPMrUrDIOF_s1azdKrH1L0KJW9Cw_NuqFspM5OsRjMpECCpwtne)
>- 23 trials, ~ two hours EEG signal/trial
>- 17 electrode channels, 200Hz sampling rate
>- This dataset is labeled by [PERCLOS level](https://iopscience.iop.org/article/10.1088/1741-2552/aa5a98/meta?casa_token=zMmqflOHEYYAAAAA:F7YusFzBVULbjWBmoy39cvGI9RPMrUrDIOF_s1azdKrH1L0KJW9Cw_NuqFspM5OsRjMpECCpwtne)
