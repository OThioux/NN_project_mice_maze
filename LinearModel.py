import numpy as np
from sklearn.linear_model import LinearRegression


def eucledian_dist(pos1, pos2):
    x = pos1[0] - pos2[0]
    y = pos1[1] - pos2[1]
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))


training_ratio = 0.1

ecephys_sep = np.load("Test_cleaned_largest_sep.npy", allow_pickle=True)
pos_data_sep = np.load("NN_projectPosDatSensor_cleaned_largest_sep.npy", allow_pickle=True)
print(pos_data_sep.shape)
print(ecephys_sep.shape)

ecephys_sep.reshape(-1, 64)
limit = int(training_ratio * len(ecephys_sep))
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
