import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Anwendungsparameter --------------------------------------------
dataset_path = '/dataset'
num_classes = 10
training_epochs = 10
# Definieren der erfassten Bildgröße
image_size = (1080 , 1920)

# Datenanpassung und Datennormierung -----------------------------
data_generator = ImageDataGenerator(
    # Datennormierung
    rescale=1./255,
    # Datenanpassung
    rotation_range=5,
    width_shift_range=20,
    height_shift_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip= True,
    validation_split=0.2
)

# Generiert Datensatz für die Trainingsdaten
train_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=32, 
    # Definition d. Batchsize für das verwendete Batchlernverfahren
    class_mode='categorical',
    subset='training'
)

# Generiert Datensatz für die Validierungsdaten
validation_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model --------------------------------------------------------
model = Sequential()
# -- convolutional Layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# -- dense Layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Definierung und Durchführung des Trainingsablaufs
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=training_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Model wird gespeichert mit angegeben Namen
model.save('og_model.h5')