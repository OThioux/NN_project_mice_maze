import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

num_folds = 9
HINDSIGHT = 2

ecephys_sep = np.load("Yuta23_data/5/NN_projectYutaMouse23_ecephys_5_cleaned_sep_fft.npy", allow_pickle=True)
pos_data_sep = np.load("Yuta23_data/5/NN_projectYutamouse23_posdata_5__cleaned_sep_fft.npy", allow_pickle=True)

pos_data = []
ecephys_data_past = []
if len(ecephys_sep) > HINDSIGHT + 1:
    for i in range(len(ecephys_sep[HINDSIGHT:])):
        ecephys_data_past.append(ecephys_sep[i:HINDSIGHT + i + 1])
        pos_data.append(pos_data_sep[HINDSIGHT + i])
ecephys_data_past = np.asarray(ecephys_data_past)
pos_data = pos_data
ecephys_sep = None
pos_data_sep = None


kfold = KFold(n_splits=num_folds)

fold_no = 1
for train, test in kfold.split(ecephys_data_past, pos_data):

    ecephys_train, ecephys_test = np.asarray([ecephys_data_past[i] for i in train]), np.asarray(
        [ecephys_data_past[i] for i in test])
    pos_train, pos_test = np.asarray([pos_data[i] for i in train]), np.asarray([pos_data[i] for i in test])

    fig, axs = plt.subplots(1,2,sharex='all', sharey='all')
    axs[0].scatter(pos_train[:, 0], pos_train[:, 1])
    axs[1].scatter(pos_test[:, 0], pos_test[:, 1], c="orange")
    axs[0].set_title('Training data for Data for fold : ' + str(fold_no))
    axs[1].set_title('Testing data for Data for fold : ' + str(fold_no))

    plt.show()

    fold_no += 1


