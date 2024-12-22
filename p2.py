# 2a
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron

X, y = make_blobs(n_samples=100, centers=2, random_state=42)
model = Perceptron(max_iter=1000, eta0=0.01, random_state=42).fit(X, y)

def plot_decision_boundary(model, X, y):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title('Perceptron Decision Boundary')
    plt.show()

plot_decision_boundary(model, X, y)

#2b

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

data = {
    'AND': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    'OR':  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    'XOR': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])),
}

for gate, (X, y) in data.items():
    perceptron = Perceptron(max_iter=10, eta0=1, random_state=42)
    perceptron.fit(X, y)
    y_pred = perceptron.predict(X)
    acc = accuracy_score(y, y_pred) * 100
    print(f"{gate} gate accuracy: {acc:.2f}%")
    print(f"Predictions: {y_pred}")
    print(f"True Labels: {y}")