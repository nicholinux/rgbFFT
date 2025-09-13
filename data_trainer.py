from tensorflow.keras import layers, models
from data_loader import load_data

def train_model():
    IMG_HEIGHT, IMG_WIDTH, train_generator, test_generator = load_data()

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


if __name__ == "__main__":
    train_model()
