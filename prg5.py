import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Normalize the data (scaling pixel values to 0 and 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Flatten the images from 28x28 into 784-dimensional vectors for DFFNN
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# Build the Deep Feed Forward Neural Network (DFFNN) model
def build_dffnn_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Output layer for 10 classes
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model and record accuracy for each epoch
def train_and_record_accuracy(model, X_train, y_train, X_test, y_test, epochs=20):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    return history

# Build the model
model = build_dffnn_model()

# Train the model and record accuracy for 20 epochs
history = train_and_record_accuracy(model, X_train, y_train, X_test, y_test, epochs=20)

# Plotting accuracy over epochs
epochs = range(1, 21)
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Test Accuracy')
plt.title('Training and Test Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Final test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
