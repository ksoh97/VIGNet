# import APIs
import numpy as np
from mne.filter import resample

# TODO: Scenario 1) training: sbj 1~9, test: sbj 10
# TODO: Scenario 2) training: sbj 1~10, test: sbj 1~10
def load_dataset(subject, fold):
    path = 'YOUR PATH'

    # Load data
    train_eeg = np.load(path + "/cv%01d_sbj%02d_train_eeg.npy" % (fold, subject))
    train_label = np.load(path + "/cv%01d_sbj%02d_train_label.npy" % (fold, subject))
    test_eeg = np.load(path + "/cv%01d_sbj%02d_test_eeg.npy" % (fold, subject))
    test_label = np.load(path + "/cv%01d_sbj%02d_test_label.npy" % (fold, subject))

    # Divide the training trials to training and validation trials for model selection.
    np.random.seed(seed=970304)
    rand_idx = np.random.permutation(train_eeg.shape[0])
    train_eeg = train_eeg[rand_idx, :, :]
    train_label = train_label[rand_idx]

    tmp = 40
    valid_eeg = train_eeg[:tmp, :, :]
    valid_label = train_label[:tmp]
    train_eeg = train_eeg[tmp:, :, :]
    train_label = train_label[tmp:]

    train_eeg, valid_eeg, test_eeg = np.expand_dims(train_eeg, -1), np.expand_dims(valid_eeg, -1), np.expand_dims(test_eeg, -1)

    # Downsampling
    train_eeg, valid_eeg, test_eeg = resample(train_eeg, up=1, down=10, axis=-2), resample(valid_eeg, up=1, down=10, axis=-2), resample(test_eeg, up=1, down=10, axis=-2)
    train_eeg, valid_eeg, test_eeg = train_eeg[:, :2, :, :], valid_eeg[:, :2, :, :], test_eeg[:, :2, :, :]
    return train_eeg, train_label, valid_eeg, valid_label, test_eeg, test_label


def load_datasetabc(subject, fold):
    path = '/home/hjkwon/Desktop/pycharm-2018.3.5/projects/Data/'

    # Load data
    train_eeg = np.load(path + "/TIME_Sess01_sub01_train.npy")
    train_label = np.load(path + "/TIME_Sess01_sub01_trlbl.npy")
    test_eeg = np.load(path + "/TIME_Sess01_sub01_test.npy")
    test_label = np.load(path + "/TIME_Sess01_sub01_tslbl.npy")

    train_eeg, test_eeg = np.moveaxis(train_eeg, -1, 0), np.moveaxis(test_eeg, -1, 0)
    print(train_eeg.shape, test_eeg.shape)

    # Divide the training trials to training and validation trials for model selection.
    np.random.seed(seed=970304)
    rand_idx = np.random.permutation(train_eeg.shape[0])
    train_eeg = train_eeg[rand_idx, :, :]
    train_label = train_label.T
    test_label = test_label.T

    train_label = train_label[rand_idx, :]

    tmp = 10
    valid_eeg = train_eeg[:tmp, :, :]
    valid_label = train_label[:tmp, :]
    train_eeg = train_eeg[tmp:, :, :]
    train_label = train_label[tmp:, :]

    train_eeg, valid_eeg, test_eeg = np.expand_dims(train_eeg, -1), np.expand_dims(valid_eeg, -1), np.expand_dims(test_eeg, -1)

    train_label = np.argmax(train_label, axis=-1)
    valid_label = np.argmax(valid_label, axis=-1)
    test_label = np.argmax(test_label, axis=-1)
    # tmp = np.argmax(train_label, axis=-1)
    # print(tmp)

    # Downsampling
    # train_eeg, valid_eeg, test_eeg = resample(train_eeg, up=1, down=10, axis=-2), resample(valid_eeg, up=1, down=10, axis=-2), resample(test_eeg, up=1, down=10, axis=-2)
    # train_eeg, valid_eeg, test_eeg = train_eeg[:, :2, :, :], valid_eeg[:, :2, :, :], test_eeg[:, :2, :, :]
    return train_eeg, train_label, valid_eeg, valid_label, test_eeg, test_label

a, b, c, d, e, f = load_datasetabc(1, 1)
print(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape)
#
# tmp = e[21, :, :, :]
#
# for i in range(a.shape[0]):
#     print(i)
#     if np.sum(a[i, :, :, :] - tmp) == 0: print("hey")

# TODO: Scenario 1) training: sbj 1~9, test: sbj 10
def load_dataset_scen1(subject, fold):
    path = 'YOUR PATH'

