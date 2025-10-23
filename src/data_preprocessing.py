

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_iris_binary():
    """
    Load and filter the Iris dataset to keep only two classes (Setosa, Versicolor).
    Returns: X, y, feature_names
    """
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # first two features for visualization
    y = iris.target
    mask = y < 2
    X = X[mask]
    y = y[mask]
    feature_names = iris.feature_names[:2]
    return X, y, feature_names


def visualize_data(X, y, feature_names, save_path="figures/iris_distribution.png"):
    """
    Scatter plot of the two Iris classes for visualization.
    """
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Setosa")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="Versicolor")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Iris Feature Distribution")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Normalize and split dataset into train/test sets.
    Returns: x_train, x_test, y_train, y_test
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def display_sample(X, y, feature_names, n=10):
    """
    Display a table of the first n samples.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    print("🔍 Sample data preview:")
    return df.head(n)
