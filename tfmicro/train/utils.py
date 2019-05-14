"""Generate simulated data for benchmarking a fully-connected or convolutional
neural net. This data is used for training the neural network to classify
a bunch of if statements. Allows us to easily create dummy TF-Micro models
to check inference time and power consumption.
"""
import numpy as np
import random
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def generate_simulated_data(datasize=1000000, balance=True, verbose=False):
    # We will first "fake" our data and try to learn a function using this
    # faked function before we try using AIRSIM.
    FAKED_TRAINING_DATA = datasize 
    x_data, y_data = [], []
    for _ in range(FAKED_TRAINING_DATA):
        faked_inputs = np.array([np.random.random() * 200 for _ in range(4)])
        xydirs = faked_inputs[:]
        
        # Make default policy just staying still
        policy = 0
        # Get the closest distance
        min_dis = xydirs.min()
        # Get the two largest distances! If the two largest distances are next
        # to each other, escape diagonally randomly. Otherwise, just escape in the largest
        # dimension
        largest1, largest2 = (-xydirs).argsort()[:2]
        max_dirs = set([largest1, largest2])
        if min_dis < 20:   #: If something is within 20 cm, let's avoid!
            if 0 in max_dirs and 1 in max_dirs: 
                policy = np.random.choice([1, 2, 3])
            elif 1 in max_dirs and 2 in max_dirs: 
                policy = np.random.choice([3, 4, 5])
            elif 2 in max_dirs and 3 in max_dirs: 
                policy = np.random.choice([5, 6, 7])
            elif 3 in max_dirs and 0 in max_dirs: 
                policy = np.random.choice([7, 8, 1])
            # If the twwoi max directions arent adjacent, just choose the largest one
            elif largest1 == 0:
            # if largest1 == 0:
                policy = 1
            elif largest1 == 1:
                policy = 3
            elif largest1 == 2:
                policy = 5
            elif largest1 == 3:
                policy = 7

        if balance and policy == 0 and random.random() > 0.1:
            continue
        x_data.append(faked_inputs)
        y_data.append(policy)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    if verbose:
        plt.hist(y_data)
        plt.title('Distribution of policies for randomly generated data')
        plt.show()
    print("Shape of x_data: {}, y_data: {}".format(x_data.shape, y_data.shape))

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2)
    return x_train, y_train, x_test, y_test
    