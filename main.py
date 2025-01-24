import subprocess
import sys

reqLibraries = [
    "tensorflow",
    "numpy",
    "Pillow",
    "matplotlib"
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
import matplotlib.pyplot as plt

# load the MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the images to range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

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

# Train the model
history = model.fit(
    x_train,        # Training Images/Input Data
    y_train,        # One-hot encoded training labels
    epochs=5,       # Number of times the model will go through the entire dataset
    batch_size=32,  # Number of samples processed before updating weights
    validation_data=(x_test, y_test)    # Validation set for monitoring performance
)


# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# Make Predictions
predictions = model.predict(x_test)
predicted_label = predictions[57].argmax()   # get the class with the highest probability


# Display the first test image and the model's prediction
plt.imshow(x_test[57], cmap='gray')
plt.title(f"Predicted Label: {predicted_label}")
plt.show()


# Make Digit Recognizer App
class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master