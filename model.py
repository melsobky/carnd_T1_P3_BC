import csv
import cv2
import numpy as np
import os
from random import shuffle
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential
from keras.layers import Flatten,Dense,Conv2D,Activation,MaxPooling2D,Dropout,Lambda,Cropping2D

"""
this function takes the path to the training data sv file and returns to samples sets
"""
def load_data_samples(datapath):
    samples = []
    with open(datapath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

"""
this function is a generator that yield a batch of the training data
"""
def generator(samples, batch_size=32):
    correction_factor = 0.2
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # load the center camera image
                c_image = cv2.imread(batch_sample[0])
                # load the left camera image
                l_image = cv2.imread(batch_sample[1])
                # load the right camera image
                r_image = cv2.imread(batch_sample[2])

                # augment the images by flipping them horizontally
                c_image_flipped = np.fliplr(c_image)
                l_image_flipped = np.fliplr(l_image)
                r_image_flipped = np.fliplr(r_image)

                # load the steering angle for the center image
                c_measurement = float(batch_sample[3])
                # add a correction factor for the left and right steering angles
                l_measurement = c_measurement + correction_factor
                r_measurement = c_measurement - correction_factor

                # augment the angels by taking there opposite values
                c_measurement_flipped = - c_measurement
                l_measurement_flipped = - l_measurement
                r_measurement_flipped = - r_measurement

                #add the images to the image list
                images.append(c_image)
                images.append(l_image)
                images.append(r_image)
                images.append(c_image_flipped)
                images.append(l_image_flipped)
                images.append(r_image_flipped)

                #add the angles to the angles list with the same order as the images
                angles.append(c_measurement)
                angles.append(l_measurement)
                angles.append(r_measurement)
                angles.append(c_measurement_flipped)
                angles.append(l_measurement_flipped)
                angles.append(r_measurement_flipped)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_samples, validation_samples = load_data_samples('../data/Long2/driving_log.csv')

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# I used the Nvidia architecture and added a dropout layer to prevent over fitting
model = Sequential()
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x:x/255.0 - 0.5))
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compile and train the model using the generator function
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=2)

model.save('model.h5')




