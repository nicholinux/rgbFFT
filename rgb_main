# data_trainer.py
"""
Combined data loader and trainer for RGB FFT Research project.

"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
train_dir = r'C:\\Users\\WimmLab\\OneDrive - Georgia Southern University (1)\\Thesis\\Dataset\\RGBFFT\\Real_AI_SD_LD_Dataset\\train'
test_dir = r'C:\\Users\\WimmLab\\OneDrive - Georgia Southern University (1)\\Thesis\\Dataset\\RGBFFT\\Real_AI_SD_LD_Dataset\\test'


def load_data():
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False   # important for confusion matrix
    )

    print("Train and test generators are ready.")
    return IMG_HEIGHT, IMG_WIDTH, train_generator, test_generator


def train_model():
    IMG_HEIGHT, IMG_WIDTH, train_generator, test_generator = load_data()

    # ----- Build model -----
    model = models.Sequential([
        layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # ----- Compute class weights -----
    labels = train_generator.classes
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights_array))

    print("Computed class weights:")
    for k, v in class_weights.items():
        print(f"Class {k}: {v:.4f}")

    # ----- EarlyStopping -----  
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3,         
        min_delta=0.140,     
        mode='min',          
        restore_best_weights=True 
    )


    # ----- Train the model -----
    print("Starting training...")
    model.fit(
        train_generator,
        validation_data=test_generator,
        class_weight=class_weights,
        callbacks=[early_stopping],
        epochs=10
    )
    print("Training complete.")

    # ----- Generate confusion matrix -----
    print("Generating confusion matrix...")
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    target_names = list(test_generator.class_indices.keys())

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ----- Plot confusion matrix heatmap -----
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm, annot=False, fmt="d", cmap="YlGnBu",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title("Confusion Matrix Heatmap", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save and show
    heatmap_path = "confusion_matrix.png"
    plt.savefig(heatmap_path, dpi=300)
    print(f"Confusion matrix heatmap saved to {heatmap_path}")

    plt.show()


if __name__ == "__main__":
    train_model()
