import numpy as np
from sklearn.linear_model import LinearRegression


def eucledian_dist(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))

# THIS IS NOT USED IN THE FINAL REPORT.
training_ratio = 0.1
training_data = int(60 * 32 * 19.5)

ecephys_sep = np.load("Yuta23_data/4/NN_project_Yutamouse23_4_ecephys_cleaned_sep_fft.npy", allow_pickle=True)
pos_data_sep = np.load("Yuta23_data/4/NN_project_Yutamouse23_4_posdata_cleaned_sep_fft.npy", allow_pickle=True)
ecephys_sep = ecephys_sep
pos_data_sep = pos_data_sep
print(pos_data_sep.shape)
print(ecephys_sep.shape)

x, y, z = ecephys_sep.shape
ecephys_sep = ecephys_sep.reshape(x, y*z)
limit = training_data
print("Limit == " + str(limit))

training_ecephys = ecephys_sep[:limit]
training_pos = pos_data_sep[:limit]
testing_ecephys = ecephys_sep[limit:]
testing_pos = pos_data_sep[limit:]
print("Data split: Training - " + str(training_ecephys.shape) + " Testing - " + str(testing_ecephys.shape))
print("                       " + str(training_pos.shape) + "            " + str(testing_pos.shape))

model = LinearRegression()

model.fit(training_ecephys, training_pos)

print("Training data: ")
pred = model.predict(training_ecephys)
dist = []
for i in range(len(training_ecephys)):
    dist.append(eucledian_dist(pred[i], training_pos[i]))
print("Results : " + str(np.mean(dist)))
print("Confint: " + str(np.percentile(dist, [2.5, 97.5])))

print("Testing data: ")
dist = []
pred = model.predict(testing_ecephys)
for i in range(len(testing_ecephys)):
    dist.append(eucledian_dist(pred[i], testing_pos[i]))

print("Results : " + str(np.mean(dist)))
print("Confint: " + str(np.percentile(dist, [2.5, 97.5])))

