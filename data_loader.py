# data_loader.py
"""
Loads training and test images from Google Drive using Keras ImageDataGenerator.
Assumes usage in Google Colab.
"""

import os
from google.colab import drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Set paths
train_dir = '/content/drive/MyDrive/RGB FFT Research/train'
test_dir = '/content/drive/MyDrive/RGB FFT Research/test'

# Image parameters
IMG_HEIGHT = 224  # Change if needed
IMG_WIDTH = 224   # Change if needed
BATCH_SIZE = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Change to 'binary' if 2 classes
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Change to 'binary' if 2 classes
)

print("Train and test generators are ready.")
from tensorflow.keras import layers, models

# Sequential CNN model
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10  
)
print("Training complete.")
