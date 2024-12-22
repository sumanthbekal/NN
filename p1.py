import numpy as np
import matplotlib.pyplot as plt

def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

x = np.linspace(-10, 10, 400)
y = {
    'sigmoid': 1 / (1 + np.exp(-x)),
    'tanh': np.tanh(x),
    'relu': np.maximum(0, x),
    'softmax': softmax(np.array([x, x * 0.5, x * 0.2])).T
}

titles = ["Sigmoid", "Tanh", "ReLU", "Softmax"]
colors, markers = ['blue', 'orange', 'green'], ['-', '--', '-.']

plt.figure(figsize=(10, 6))
for i, (key, y_values) in enumerate(y.items()):
    plt.subplot(2, 2, i + 1)
    if key == 'softmax':
        for j in range(y_values.shape[1]):
            plt.plot(x, y_values[:, j], label=f"Set {j+1}", linestyle=markers[j])
    else:
        plt.plot(x, y_values, color=colors[i], label=key)
    plt.title(titles[i])
    plt.grid(); plt.legend()

plt.tight_layout()
plt.show()