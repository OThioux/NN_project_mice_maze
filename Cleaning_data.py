import numpy as np
import math

sensor0 = np.load("NN_projectPosDatSensor0.npy")
sensor1 = np.load("NN_projectPosDatSensor1.npy")
SENSOR_FREQ = 39
ECEPHYS = 1250
ecephys = np.load("Test.npy")


t_pos = 1/39
t_ece = 1/1250

i = 0

while math.isnan(sensor0[i, 0]):
    i += 1

while math.isnan(sensor1[i, 0]):
    i += 1

sensor0 = sensor0[i:]
sensor1 = sensor1[i:]

t = i * t_pos
j = t/t_ece
ecephys = ecephys[j:]
print(ecephys[0])
