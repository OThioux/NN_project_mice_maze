import matplotlib.pyplot as plt
import numpy as np

pos_dat = np.load("/Yuta23_data")
y = pos_dat[1000:2000, 1]
x = pos_dat[1000:2000, 0]
plt.scatter(x, y)
plt.ylim([0,500])
plt.xlim([50,600])
plt.show()
