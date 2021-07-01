import numpy as np
import matplotlib.pyplot as plt

# plots the data from the electrophysiology

no_plots = 10
no_points = 500

ecephys_sep = np.load("Yuta23_data/5/NN_projectYutaMouse23_ecephys_5.npy", allow_pickle=True)

print(ecephys_sep.shape)
ecephys_sep = ecephys_sep[2000:2000 + no_points, 0:no_plots]
print(ecephys_sep.shape)
# print(ecephys_sep[:, 0].shape)

fig, axs = plt.subplots(no_plots, 1, sharex='all', sharey='all')
for i in range(len(axs)):
    print("Plotting : " + str(i))
    axs[i].plot(np.arange(no_points), ecephys_sep[:, i])

plt.show()


