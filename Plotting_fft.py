import numpy as np
import matplotlib.pyplot as plt

# plots the data from the fourier transform

no_plots = 4

ecephys_sep = np.load("Yuta23_data/5/NN_projectYutaMouse23_ecephys_5_cleaned_sep_fft.npy", allow_pickle=True)

fig, axs = plt.subplots(no_plots, 1, sharex='all', sharey='all')
for i in range(no_plots):
    axs[i].plot(ecephys_sep[1000, 1:, i])

plt.show()
