from tensorflow import keras
from keras import regularizers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
from keras import backend as K
import Largest_smallest_diff

EPOCHS = 5
BATCHSIZE = 25
HINDSIGHT = 4
TRAINING_DATA = int(60 * 32 * 19.5)


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# loading data
ecephys_sep = np.load("Yuta23_data/4/NN_project_Yutamouse23_4_ecephys_cleaned_sep_fft.npy", allow_pickle=True)
pos_data_sep = np.load("Yuta23_data/4/NN_project_Yutamouse23_4_posdata_cleaned_sep_fft.npy", allow_pickle=True)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

training_ecephys = np.asarray(ecephys_data_past[:TRAINING_DATA])
training_pos = np.asarray(pos_data[:TRAINING_DATA])
testing_ecephys = np.asarray(ecephys_data_past[TRAINING_DATA:])
testing_pos = np.asarray(pos_data[TRAINING_DATA:])
print("Data split: Training - " + str(training_ecephys.shape) + " Testing - " + str(testing_ecephys.shape))
print("                       " + str(training_pos.shape) + "          " + str(testing_pos.shape))

# Plotting the data
plt.scatter(training_pos[:, 0], training_pos[:, 1])
plt.scatter(testing_pos[:, 0], testing_pos[:, 1])
plt.title('Mouse positions for training data (mm)')
plt.legend(['test', 'train'], loc='upper right')
plt.ylabel('Y')
plt.xlabel('X')
plt.ylim([0, 500])
plt.xlim([50, 600])
plt.show()
plt.savefig("training_data_plotted.png")

model = keras.models.Sequential()

# hidden layer, directly follows from input, 100 neurons, input vector of dim 64
# model.add(keras.layers.Dense(100, input_shape=ecephys_data_past[0].shape, activation="tanh"))
model.add(keras.layers.Flatten(input_shape=ecephys_data_past[0].shape))
# model.add(keras.layers.Dropout(.01))
model.add(keras.layers.Dense(100, activation="tanh", kernel_regularizer=regularizers.l1(0.0000001), use_bias=True))
model.add(keras.layers.Dense(50, activation="tanh"))
# model.add(keras.layers.Dropout(.2))
# output layer, vector of dim 2, position
model.add(keras.layers.Dense(2, activation="tanh"))
model.add(keras.layers.Dense(2))

model.summary()

model.compile(optimizer="SGD", loss=euclidean_distance_loss)

K.set_value(model.optimizer.learning_rate, 1.0)
hist = model.fit(training_ecephys, training_pos, epochs=EPOCHS, batch_size=BATCHSIZE,
                 validation_data=(testing_ecephys, testing_pos))
loss = hist.history['loss']
val_loss = hist.history['val_loss']
K.set_value(model.optimizer.learning_rate, .1)
hist = model.fit(training_ecephys, training_pos, epochs=EPOCHS, batch_size=BATCHSIZE,
                 validation_data=(testing_ecephys, testing_pos))
loss = np.concatenate((loss, hist.history["loss"]))
val_loss = np.concatenate((val_loss, hist.history["val_loss"]))
# from tutorial
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.savefig("FNN_model_plot_little_training_5min_perceptron.png")

print("Testing data: ")

pred = model.predict(testing_ecephys)
dist = Largest_smallest_diff.eucledian_dist(pred, testing_pos)
# for i in range(len(testing_ecephys)):
#     dist.append(euclidean_distance_loss(pred[i], testing_pos[i]))

print("Results : " + str(np.mean(dist)))
print("Confint: " + str(np.percentile(dist, [2.5, 97.5])))
mean_dist = Largest_smallest_diff.get_random_chance(testing_pos, training_pos)

model.save("FNN_Model_little_training_5min_perceptron.h5")

print("Done")
