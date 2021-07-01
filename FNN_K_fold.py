from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras import backend as K
import Largest_smallest_diff



EPOCHS = 1
BATCHSIZE = 25
HINDSIGHT = 2
num_folds = 9


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


# loading data
ecephys_sep = np.load("Yuta23_data/5/NN_projectYutaMouse23_ecephys_5_cleaned_sep_fft.npy", allow_pickle=True)
pos_data_sep = np.load("Yuta23_data/5/NN_projectYutamouse23_posdata_5__cleaned_sep_fft.npy", allow_pickle=True)

# Making sure that Tensorflow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if (tf.test.gpu_device_name):
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))

# Adding hindsight to the input data for the models
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

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds)

# Bunch of variables for saving.
plt_lines = int(np.sqrt(num_folds) + 0.5)
fig, axs = plt.subplots(plt_lines, int(num_folds/plt_lines + 0.5))
fold_no = 1
hist = []
loss = []
loss_hist = []
guess_loss_hist = []
loss_hist_conf = [[], []]
linear_loss_hist_conf = [[], []]
linear_loss_hist = []
for train, test in kfold.split(ecephys_data_past, pos_data):
    model = keras.models.Sequential()

    # Get the training and testing data.
    ecephys_train, ecephys_test = np.asarray([ecephys_data_past[i] for i in train]), np.asarray(
        [ecephys_data_past[i] for i in test])
    pos_train, pos_test = np.asarray([pos_data[i] for i in train]), np.asarray([pos_data[i] for i in test])

    # Plotting the data
    c_axs = axs[int((fold_no - 1)/plt_lines), (fold_no - 1)%plt_lines]
    c_axs.scatter(pos_train[:, 0] , pos_train[:, 1])
    c_axs.scatter(pos_test[:, 0], pos_test[:, 1])
    c_axs.set_title('Data for fold : ' + str(fold_no))
    c_axs.legend(['train', 'test'], loc='upper right')

    # Hyper-parameters are described in the paper in further detail.
    # First layer is simply to flatten the data.
    model.add(keras.layers.Flatten(input_shape=ecephys_data_past[0].shape))
    # second layer is a dense layer with 25 neurons.
    model.add(keras.layers.Dense(25, activation="tanh", kernel_regularizer=regularizers.l1(5.0e-7), use_bias=True,
    kernel_initializer="glorot_uniform"))
    # Third layer is a dense layer with 10 neurons.
    model.add(keras.layers.Dense(10, activation="tanh", kernel_regularizer=regularizers.l1(5.0e-7), use_bias=True,
    kernel_initializer="glorot_uniform"))
    # output layer, vector of dim 2, position
    model.add(keras.layers.Dense(2))

    model.summary()
    print('------------------------------------------------------------------------')

    print(f'Training for fold {fold_no} ...')
    # Shows us how the data is split, we just want to make sure that this always stays the same for all folds.
    print("Data split: Training - " + str(ecephys_train.shape) + " Testing - " + str(ecephys_test.shape))
    print("                       " + str(pos_train.shape) + "          " + str(pos_test.shape))
    # Compile the model using the personalized loss function
    model.compile(optimizer="adam", loss=euclidean_distance_loss)

    # Start fitting the model with learning rate 1.0 for 1 epoch
    K.set_value(model.optimizer.learning_rate, 1.0)
    loss = model.fit(ecephys_train, pos_train, epochs=EPOCHS, batch_size=BATCHSIZE,
                     validation_data=(ecephys_test, pos_test))
    # Star fitting the model with learning rate 0.1 for 2 epochs
    K.set_value(model.optimizer.learning_rate, .1)
    hist.append(model.fit(ecephys_train, pos_train, epochs=EPOCHS + 1, batch_size=BATCHSIZE,
                          validation_data=(ecephys_test, pos_test)))

    # Print the stats of the model on the testing data.
    print("Testing data: ")
    pred = model.predict(ecephys_test)
    dist = Largest_smallest_diff.eucledian_dist(pred, pos_test)
    mean_dist = np.mean(dist)
    loss_hist.append(mean_dist)
    print("Results : " + str(mean_dist))
    confint = np.percentile(dist, [15.9, 84.1])
    loss_hist_conf[0].append(mean_dist - confint[0])
    loss_hist_conf[1].append(confint[1] - mean_dist)
    print("Confint: " + str(confint))
    mean_dist_guess = Largest_smallest_diff.get_random_chance(pos_train, pos_test)
    guess_loss_hist.append(mean_dist_guess)

    # Train and test the linear model.
    print("\nLinear model:")
    print(ecephys_train[:, HINDSIGHT-1].shape)
    # Remove all the hindsight from the linear model such that it is a naive as possible.
    x, y, z = ecephys_train[:, HINDSIGHT-1].shape
    ecephys_train = ecephys_train[:, HINDSIGHT-1].reshape(x, y * z)
    x, y, z = ecephys_test[:, HINDSIGHT-1].shape
    ecephys_test = ecephys_test[:, HINDSIGHT-1].reshape(x, y * z)

    # Train the linear model.
    model_linear = LinearRegression()
    model_linear.fit(ecephys_train, pos_train)

    # Print the outcome.
    print("Testing data: ")
    pred = model_linear.predict(ecephys_test)
    dist_linear = Largest_smallest_diff.eucledian_dist(pred, pos_test)
    linear_mean_dist = np.mean(dist_linear)
    linear_loss_hist.append(linear_mean_dist)
    print("Results : " + str(linear_mean_dist))
    confint_linear = np.percentile(dist_linear, [15.9, 84.1])
    print("Confint: " + str(confint_linear))
    linear_loss_hist_conf[0].append(linear_mean_dist - confint_linear[0])
    linear_loss_hist_conf[1].append(confint_linear[1] - linear_mean_dist)

    fold_no += 1

print("Done")

# Show what the data looks like
plt.show()

# Show our beautiful k-fold plot.
width = 0.3
pos = np.array(range(num_folds))
plt.bar(pos - width, loss_hist, width, yerr=loss_hist_conf, capsize=7)
plt.bar(pos, guess_loss_hist, width)
plt.bar(pos + width, linear_loss_hist, width, yerr=linear_loss_hist_conf, capsize=7)

plt.title('Models')
plt.ylabel('loss')
plt.xlabel('Fold')
plt.legend(['Loss', 'Educated guess loss', 'Linear model loss'], loc='upper right')
plt.show()
plt.savefig("FNN_k_fold.png")

# Save the model, this won't actually work but whatever.
model.save("FFN_Model_smol_DistributedSampling.h5")
