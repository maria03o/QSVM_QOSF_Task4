

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_decision_boundary(model, quantum_kernel, x_data, y_data, x_train, title, save_path):
    """
    Plot QSVM decision boundary for 2D data.
    """
    os.makedirs("figures", exist_ok=True)

    xx, yy = np.meshgrid(
        np.linspace(x_train[:, 0].min() - 1, x_train[:, 0].max() + 1, 100),
        np.linspace(x_train[:, 1].min() - 1, x_train[:, 1].max() + 1, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    kernel_grid = quantum_kernel.evaluate(x_vec=grid_points, y_vec=x_train)
    Z = model.predict(kernel_grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    plt.scatter(x_data[y_data == 0, 0], x_data[y_data == 0, 1], color="blue", label="Setosa")
    plt.scatter(x_data[y_data == 1, 0], x_data[y_data == 1, 1], color="red", label="Versicolor")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def compare_performance(results, save_path="figures/performance_comparison.png"):
    """
    Create a bar chart comparing accuracy across models.
    """
    os.makedirs("figures", exist_ok=True)

    models = list(results.keys())
    train_acc = [results[m]["train_acc"] * 100 for m in models]
    test_acc = [results[m]["test_acc"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, train_acc, width, label="Train")
    plt.bar(x + width/2, test_acc, width, label="Test")
    plt.xticks(x, models)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
