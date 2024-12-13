import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.regularizers import l1, l2

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the data to fit the model (since Fashion MNIST is grayscale, we add a channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Function to build the base model
def build_base_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to build the L1 regularized model
def build_l1_regularized_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l1(0.01)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l1(0.01)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l1(0.01)),
        layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l1(0.01)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=l1(0.01)),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to build the L2 regularized model
def build_l2_regularized_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.01)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to build the model with Dropout
def build_dropout_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Adding dropout with a rate of 0.5
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc

# Build and evaluate base model
print("Training Base Model")
base_model = build_base_model()
base_acc = train_and_evaluate_model(base_model, X_train, y_train, X_test, y_test)
print(f"Base Model Test Accuracy: {base_acc:.4f}")

# Build and evaluate L1 regularized model
print("\nTraining L1 Regularized Model")
l1_model = build_l1_regularized_model()
l1_acc = train_and_evaluate_model(l1_model, X_train, y_train, X_test, y_test)
print(f"L1 Regularized Model Test Accuracy: {l1_acc:.4f}")

# Build and evaluate L2 regularized model
print("\nTraining L2 Regularized Model")
l2_model = build_l2_regularized_model()
l2_acc = train_and_evaluate_model(l2_model, X_train, y_train, X_test, y_test)
print(f"L2 Regularized Model Test Accuracy: {l2_acc:.4f}")

# Build and evaluate Dropout model
print("\nTraining Dropout Model")
dropout_model = build_dropout_model()
dropout_acc = train_and_evaluate_model(dropout_model, X_train, y_train, X_test, y_test)
print(f"Dropout Model Test Accuracy: {dropout_acc:.4f}")