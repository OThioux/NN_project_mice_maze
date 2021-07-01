import numpy as np


# Gets the educated guess loss.
def get_random_chance(points, training_data):
    points = np.asarray(points)
    mean_point = np.asarray(training_data).mean(axis=0)
    distances = eucledian_dist(points, [mean_point] * len(points))
    avg_distance = np.asarray(distances).mean()
    print("Guessing: " + str(avg_distance))
    print("Confint: " + str(np.percentile(distances, [2.5, 97.5])))
    return avg_distance


# Loss function.
def eucledian_dist(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis=-1))


