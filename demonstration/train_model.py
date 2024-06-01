import os
import random
import glob
import numpy as np
import cv2

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

import matplotlib.pyplot as plt


# Random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset path
DATASET_PATH = "../data/hand_gesture_recog/data"

MAX_SAMPLE_PER_GESTURE = 1600

images = []
labels = []
label_to_class = {}

gestures_paths = glob.glob(os.path.join(DATASET_PATH, "*"))
gestures_paths.sort()
print(gestures_paths)

for i, gestures_path in enumerate(gestures_paths):
    gesture_type = gestures_path.split("/")[-1]
    gestures = glob.glob(os.path.join(gestures_path, "*"))
    label = [0] * len(gestures_paths)
    label[i] = 1
    label_to_class[i] = gesture_type
    for j, gesture in enumerate(gestures):
        if j == MAX_SAMPLE_PER_GESTURE:
            break
        img = cv2.imread(gesture)
        img = cv2.resize(img,(100, 100))            # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Gray scale
        images.append(img)
        labels.append(label)

# Model
model = Sequential()

# First conv layer
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(100, 100, 1))) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second conv layer
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fully connected layer
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

# Softmax layer
model.add(Dense(6, activation="softmax"))

# Model summary
optimiser = Adam()
model.compile(optimizer=optimiser, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
model.summary()

X = np.asarray(images)
y = np.asarray(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=4)

X_train = X_train.reshape(X_train.shape[0], 100, 120, 1)
X_test = X_test.reshape(X_test.shape[0], 100, 120, 1)

model.fit(X_train,
          y_train,
          batch_size=64,
          epochs=128,
          verbose=1,
          validation_data=(X_test, y_test))

model.save("server_model.keras")