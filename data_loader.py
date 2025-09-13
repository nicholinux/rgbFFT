# data_loader.py
"""
Loads training and test images from Google Drive using Keras ImageDataGenerator.

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

def load_data():
    IMG_HEIGHT = 224  
    IMG_WIDTH = 224   
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary' 
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary' 
    )

    print("Train and test generators are ready.")
    return IMG_HEIGHT, IMG_WIDTH, train_generator, test_generator


if __name__ == "__main__":
    load_data()
