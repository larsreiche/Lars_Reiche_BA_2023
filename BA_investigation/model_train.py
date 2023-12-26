import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Verwendeter Datensatz während eines Trainingszyklus
dataset_path = '/datasets/kampos0'
# dataset_path = '/datasets/kampos1'
# dataset_path = '/datasets/kampos2'

# Anzahl der verwendeten Klassen
num_classes = 10

# Definition der Bildgröße
image_size = (640, 480)

# Datenanpassung und Datennormierung -----------------------------
data_generator = ImageDataGenerator(
    # Datennormierung
    rescale=1./255,
    # Datenanpassung
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Generiert Datensatz für die Trainingsdaten
train_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=32,
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
model.add(Flatten())
# -- dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Definierung und Durchführung des Trainingsablaufs
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Erstellen der Graphen über die Qualitätsmaße über den Trainingsverlauf
# Quelaitätsmaß - Trainingsgenauigkeit
plt.plot(model.history.history['accuracy'])
plt.title('Trainingsgenauigkeit')
plt.xlabel('Epochen')
plt.ylabel('Genauigkeit')
plt.show()
# Quelaitätsmaß - Validierungsgenauigkeit
plt.plot(model.history.history['val_accuracy'])
plt.title('Validierungsgenauigkeit')
plt.xlabel('Epochen')
plt.ylabel('Genauigkeit')
plt.show()

# Speichern des Models
model.save('og6.h5')