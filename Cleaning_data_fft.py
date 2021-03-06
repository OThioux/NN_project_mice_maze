import sys

import numpy as np

# Definition of the euclidean_distance loss function
def euclidean_dist(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))


# Calculates the mean distance between all the points. We used this to get an idea of the spread of our data.
def mean_dist(points):
    points = np.asarray(points)
    mean_point = points[:60*10*39].mean(axis=0)
    distances = euclidean_dist(points, [mean_point] * len(points))
    avg_distance = np.asarray(distances).mean()
    return avg_distance


np.set_printoptions(threshold=sys.maxsize)

SENSOR_DATA_NAME = "Yuta23_data/5/NN_projectYutamouse23_posdata_5_"
ECEPHYS_DATA_NAME = "Yuta23_data/5/NN_projectYutaMouse23_ecephys_5"

# Load both sensor 0 and 1
sensor0 = np.load(SENSOR_DATA_NAME + "0.npy")
sensor1 = np.load(SENSOR_DATA_NAME + "1.npy")
# These are the sensor frequencies as defined by the database.
SENSOR_FREQ = 39.0625
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

sample_size = len(sensor0)
ecephys_clean = [[]]
pos_clean = [[]]
idx = 0
nan_intervals = 0
last_n = 0
for n in range(int(len(sensor0)/2)):
    n = n * 2
    # Because the electrophysiology data is effectively 1 time sample in the past we take the interval between
    # the sample n and n+1
    sample_time = (n + 1) * t_pos
    # The starting point of the interval is one data-point in the past, because we want 65 data-points not 32-33
    interval_start = (n - 1) * t_pos
    # Calculate the starting and ending samples.
    start_sample = int(interval_start / t_ece)
    end_sample = int(sample_time / t_ece)
    # Get the interval from the electrophysiology data.
    interval = np.asarray(ecephys[start_sample:end_sample])
    if n % 10000 == 0:
        print("Dealing with sample " + str(int(n / 1000)) + "," + "000" + " of " + str(sample_size))

    # If we do not find any missing values we continue appending to this list of uninterrupted data.
    if not np.isnan(interval).any() and not np.isnan(sensor0[n]).any() and not np.isnan(sensor1[n]).any() \
            and len(interval) >= 32:
        # Calculate the interval fft, only taking the first 12 bins.
        interval_fft = np.real(np.fft.fft(interval, n=12, axis=0))
        ecephys_clean[idx].append(interval_fft)
        pos_clean[idx].append(np.nanmean([sensor0[n], sensor1[n]], axis=0))
        last_n = n
    # If we find more than 5 missing values then we start a new list of uninterrupted data.
    elif not pos_clean[idx] == [] and n - last_n > 5:
        ecephys_clean.append([])
        pos_clean.append([])
        idx += 1

sensor0 = None
sensor1 = None
ecephys = None

# Remove excess nan results at the end, this shouldn't be necessary but we will keep it here to be sure.
i = 1
while np.isnan(ecephys_clean[-i]).any():
    i += 1

ecephys_clean = np.asarray(ecephys_clean[:-i], dtype=object)
pos_clean = pos_clean[:-i]

# Take the largest uninterrupted data
largest = 0
largest_idx = 0
for i in range(len(ecephys_clean)):
    print("For i in " + str(ecephys_clean.shape) + " len == " + str(len(ecephys_clean[i])), end="")
    dist_mean = mean_dist(pos_clean[i])
    print(" Mean distance = " + str(dist_mean))
    if 300 > dist_mean > largest and len(ecephys_clean[i]) > 80000:
        largest = dist_mean
        largest_idx = i

pos_clean = np.asarray(pos_clean[largest_idx])
ecephys_clean = np.asarray(ecephys_clean[largest_idx])

# Save the data
print("Saving...")
np.save(SENSOR_DATA_NAME + "_cleaned_sep_fft.npy", pos_clean, allow_pickle=True)
np.save(ECEPHYS_DATA_NAME + "_cleaned_sep_fft.npy", ecephys_clean, allow_pickle=True)

# Print stats
print("Position data:")
print(pos_clean.shape)
print("Electrophysiology data:")
print(ecephys_clean.shape)
print("Mean distance:")
print(largest)
