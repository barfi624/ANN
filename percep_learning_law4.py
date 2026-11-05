import numpy as np
import matplotlib.pyplot as plt

# Step function
def step(x):
    return 1 if x >= 0 else -1

# Training data (AND function - linearly separable)
X = np.array([[0,0], [1,0], [0,1], [1,1]])
Y = np.array([-1, -1, -1, 1])

# Initialize weights and bias
w = np.zeros(2)
b = 0
lr = 0.3
epochs = 10

# -----------------------------
# Perceptron Learning Law
# -----------------------------
for _ in range(epochs):
    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        net = np.dot(x, w) + b
        y_pred = step(net)

        if y_pred != y:
            w += lr * y * x
            b += lr * y

# -----------------------------
# Plot Decision Regions
# -----------------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b)
Z = Z.reshape(xx.shape)

# Plot regions
plt.contourf(xx, yy, Z, alpha=0.6)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=Y)

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Perceptron Decision Regions")
plt.show()

# Print learned parameters
print("Weights:", w)
print("Bias:", b)