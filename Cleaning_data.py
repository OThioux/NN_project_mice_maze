import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
np.warnings.simplefilter("ignore", category=RuntimeWarning)

SENSOR_DATA_NAME = "NN_projectPosDatSensor"
ECEPHYS_DATA_NAME = "Test"

# Load both sensor 0 and 1
sensor0 = np.load(SENSOR_DATA_NAME + "0.npy")
sensor1 = np.load(SENSOR_DATA_NAME + "1.npy")
# These are the sensor frequencies as defined by the database.
SENSOR_FREQ = 39
ECEPHYS_FREQ = 1250
# Load the electrophysiology data.
ecephys = np.load(ECEPHYS_DATA_NAME + ".npy")

# Calculate the times per sample for all datasets.
t_pos = 1 / SENSOR_FREQ
t_ece = 1 / ECEPHYS_FREQ

# Here we remove all the nan values from the dataset.
i = 0
while np.isnan(sensor0[i, 0]):
    i += 1
while np.isnan(sensor1[i, 0]):
    i += 1
sensor0 = sensor0[i:]
sensor1 = sensor1[i:]

# The time of the first sample is equal to the sample * the sampling time of the position sensor.
t = i * t_pos

# The sample cutoff point for the electrophysiology data starts at the time of the last point of the position data.
# This is so that we have the data leading to the point and not the data of the future.
j = int((t - t_pos) / t_ece)

# We now cut the electrophysiology data.
ecephys = ecephys[j:]

ecephys_clean = []

for n in range(len(sensor0)):
    # Because the electrophysiology data is effectively 1 time sample in the past we take the interval between
    # the sample n and n+1
    sample_time = (n + 1) * t_pos
    interval_start = n * t_pos
    # Calculate the starting and ending samples.
    start_sample = int(interval_start/t_ece)
    end_sample = int(sample_time/t_ece)
    # Calculate the interval mean over axis 1, which means per electrode we take the average over the interval.
    interval = np.asarray(ecephys[start_sample:end_sample])
    interval_mean = np.nanmean(interval, axis=0)
    ecephys_clean.append(interval_mean)


# Remove excess nan results
i = 1
while np.isnan(ecephys_clean[-i]).any():
    i += 1

ecephys_clean = np.asarray(ecephys_clean[:-i])
sensor0 = sensor0[:-i]
sensor1 = sensor1[:-i]


# Save the data

np.save(SENSOR_DATA_NAME + "0_cleaned.npy", sensor0)
np.save(SENSOR_DATA_NAME + "1_cleaned.npy", sensor1)
np.save(ECEPHYS_DATA_NAME + "_cleaned.npy", ecephys_clean)

# Print stats
print("Sensors 0 and 1:")
print(sensor0.shape)
print(sensor1.shape)
print("Electrophysiology data:")
print(ecephys_clean.shape)


