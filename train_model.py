import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
 
# Load the MNIST dataset 
mnist = keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
 
# Normalize pixel values to the range [0,1] 
x_train, x_test = x_train / 255.0, x_test / 255.0   
 
# Define a simple neural network model 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into 1D array 
    keras.layers.Dense(128, activation="relu"), 
    keras.layers.Dense(10, activation="softmax")  # Output layer for 10 digits (0-9) 
]) 
 
# Compile the model 
model.compile(optimizer="adam", 
loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
 
# Train the model 
model.fit(x_train, y_train, epochs=5) 
 
# Save the trained model 
model.save("mnist_model.h5") 
print("Model saved successfully!") 
