import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Function to generate AND, OR, XOR datasets
def generate_data(logic_type):
    if logic_type == "AND":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
    elif logic_type == "OR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
    elif logic_type == "XOR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
    else:
        raise ValueError("Unsupported logic type")
    return X, y

# Function to train and evaluate perceptron on a dataset
def train_and_evaluate(logic_type):
    print(f"\nEvaluating Perceptron on {logic_type} logic")
    
    # Generate data
    X, y = generate_data(logic_type)

    # Initialize the Perceptron model
    model = Perceptron()

    # Train the model on the entire dataset (small dataset, no train-test split)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Plot the decision boundary
    plot_decision_boundary(X, y, model, logic_type)

# Function to plot the decision boundary
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap=plt.cm.Paired)
    plt.title(f"Decision Boundary of Perceptron ({title})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Evaluate the perceptron on AND, OR, and XOR datasets
for logic in ["AND", "OR", "XOR"]:
    train_and_evaluate(logic)
