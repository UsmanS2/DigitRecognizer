import subprocess
import sys

reqLibraries = [
    "tensorflow",
    "numpy",
    "Pillow",
]

# function that installs libraries
def installLibrary(library):
    subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Check and Install Libraries
for library in reqLibraries:
    try:
        __import__(library)
        print(f"Imported {library}")
    except ImportError:
        installLibrary(library)
        print(f"Installing {library}")

# Import all necessary libraries
import tkinter as tk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# load the MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to range [0, 1]
x_train = x_train / 255.0
y_train = y_train / 255.0

# Encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# Build the model
model = Sequential([
    Flatten(input_shape=(28,28)),    # Takes the initial image and converts the 2d, 28x28, image into 1d vector
    Dense(128, activation='relu'),   # Hidden Layer, processes flattened vector, learns patterns in the data
    Dense(10, activation='softmax')  # output layer, outputs distribution across the 10 classes aka digits 0-9
])


# Compile the model
model.compile(
    optimizer='adam',  # Adaptive Moment Estimation, optimization algorithm
    loss='categorical_crossentropy',  # calculates the difference between the model's predictions and the actual labels
    metrics=['accuracy']  # used to evaluate the model's performance
)


