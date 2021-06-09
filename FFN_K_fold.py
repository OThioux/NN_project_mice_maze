from sklearn.model_selection import KFold
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
from keras import backend as K

EPOCHS = 10
BATCHSIZE = 250
HINDSIGHT = 4
num_folds = 6


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# loading data
ecephys_sep = np.load("Test_cleaned_sep.npy", allow_pickle=True)
pos_data_sep = np.load("NN_projectPosDatSensor_cleaned_sep.npy", allow_pickle=True)


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
pos_data = np.asarray(pos_data)


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

fold_no = 1
hist = []
for train, test in kfold.split(ecephys_data_past, pos_data):
    model = keras.models.Sequential()

    # hidden layer, directly follows from input, 100 neurons, input vector of dim 64
    model.add(keras.layers.Dense(100, input_shape=ecephys_data_past[0].shape, activation="tanh"))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(50, activation="tanh"))
    # model.add(keras.layers.Dropout(.2))
    # output layer, vector of dim 2, position
    model.add(keras.layers.Dense(2))

    # model.summary()
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    model.compile(optimizer="adam", loss=euclidean_distance_loss)
    # K.set_value(model.optimizer.learning_rate, 0.1)
    hist = model.fit(ecephys_data_past[train], pos_data[train], epochs=EPOCHS, batch_size=BATCHSIZE,
                     validation_data=(ecephys_data_past[test], pos_data[test]))
    fold_no += 1


print("Done")

# from tutorial
for i in range(len(hist)):
    plt.plot(hist[i].history['loss'])
    plt.plot(hist[i].history["val_loss"])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("FFN_model_plot.png")

# model.save("FFN_Model.h5")

