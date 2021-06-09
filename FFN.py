
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
from keras import backend as K

EPOCHS = 20
BATCHSIZE = 250
HINDSIGHT = 3
TESTING_DATA = 20000


# TODO: Try K-fold try fourier (higher dimensionality)

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# loading data
ecephys_sep = np.load("Test_cleaned_sep_fft.npy", allow_pickle=True)
pos_data_sep = np.load("NN_projectPosDatSensor_cleaned_sep_fft.npy", allow_pickle=True)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


pos_data = []
ecephys_data_past = []
for j in range(len(ecephys_sep)):
    if np.shape(pos_data_sep[j])[0] != np.shape(ecephys_sep[j])[0]:
        print(str(np.shape(pos_data_sep[j])) + "  " + str(np.shape(ecephys_sep[j])))
    if len(ecephys_sep[j]) > HINDSIGHT + 1:
        for i in range(len(ecephys_sep[j][HINDSIGHT:])):
            ecephys_data_past.append(ecephys_sep[j][i:HINDSIGHT + i + 1])
            pos_data.append(pos_data_sep[j][HINDSIGHT + i])
ecephys_data_past = np.asarray(ecephys_data_past)

training_ecephys = np.asarray(ecephys_data_past[:-TESTING_DATA])
training_pos = np.asarray(pos_data[:-TESTING_DATA])
testing_ecephys = np.asarray(ecephys_data_past[-TESTING_DATA:])
testing_pos = np.asarray(pos_data[-TESTING_DATA:])
print("Data split: Training - " + str(training_ecephys.shape) + " Testing - " + str(testing_ecephys.shape))
print("                       " + str(training_pos.shape) + "          " + str(testing_pos.shape))


model = keras.models.Sequential()

# hidden layer, directly follows from input, 100 neurons, input vector of dim 64
model.add(keras.layers.Dense(100, input_shape=ecephys_data_past[0].shape, activation="tanh"))
model.add(keras.layers.Flatten())
# model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Dense(50, activation="tanh"))
# model.add(keras.layers.Dropout(.2))
# output layer, vector of dim 2, position
model.add(keras.layers.Dense(2))

model.summary()

model.compile(optimizer="SGD", loss=euclidean_distance_loss)
# K.set_value(model.optimizer.learning_rate, 0.1)
hist = model.fit(training_ecephys, training_pos, epochs=EPOCHS, batch_size=BATCHSIZE,
                 validation_data=(testing_ecephys, testing_pos))

print("Done")

# from tutorial
plt.plot(hist.history['loss'])
plt.plot(hist.history["val_loss"])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("FFN_model_plot.png")

model.save("FFN_Model.h5")

