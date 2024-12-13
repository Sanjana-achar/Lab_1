import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# One-hot encode the target labels
y = to_categorical(y)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Sequential model with 4 hidden layers
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),  # Input layer + first hidden layer
    Dense(32, activation='relu'),                             # Second hidden layer
    Dense(16, activation='relu'),                             # Third hidden layer
    Dense(8, activation='relu'),                              # Fourth hidden layer
    Dense(y.shape[1], activation='softmax')                   # Output layer (softmax for multiclass)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
