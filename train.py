# Classifying Galaxies

# https://www.codecademy.com/projects/practice/classifying-galaxies-deep-learning

# Classifying Galaxies Using Convolutional Neural Networks

# This code is a machine learning program that uses convolutional neural networks (CNNs) to classify deep-space galaxies based on their image data. The data is curated by Galaxy Zoo, a crowd-sourced project devoted to annotating galaxies in support of scientific discovery. The data is split into four classes: galaxies with no identifying characteristics ([1,0,0,0]), galaxies with rings ([0,1,0,0]), galactic mergers ([0,0,1,0]), and "other," irregular celestial bodies ([0,0,0,1]).

# The program uses TensorFlow, a popular machine learning library, to build and train the CNN model.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

from visualize import visualize_activations

import app

# The data is split into training and testing sets using the train_test_split function from scikit-learn library. 

input_data, labels = load_galaxy_data()

train_test_data, test_data, train_test_labels, test_labels = train_test_split(input_data, labels, test_size=0.20, shuffle=True, random_state=222, stratify=labels)

# The ImageDataGenerator is used to preprocess the input data, rescaling it to values between 0 and 1.

data_generator = ImageDataGenerator(rescale=1./255)

training_iterator = ImageDataGenerator(rescale=1./255).flow(train_test_data, train_test_labels, batch_size=5)

label_iterator = ImageDataGenerator(rescale=1./255).flow(train_test_data, train_test_labels, batch_size=5)

# The model architecture consists of two convolutional layers followed by max-pooling layers, a flatten layer, and two fully connected layers with relu and softmax activations, respectively.

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128, 128, 3)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="softmax"))

# The model is compiled using the Adam optimizer with a learning rate of 0.001, Categorical Crossentropy loss function, and two evaluation metrics: Categorical Accuracy and AUC (Area Under the Curve).

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

model.summary()

# The model is trained on the training data using the fit function, with validation data used to evaluate the model performance. The model is trained for 8 epochs.

model.fit(training_iterator, 
          steps_per_epoch=len(train_test_data) / 5, 
          epochs=8, 
          validation_data=label_iterator, 
          validation_steps=len(test_data) / 5)

# The visualize_activations function from the visualize module is used to visualize the activations of the CNN layers on the test data.

visualize_activations(model,label_iterator)
