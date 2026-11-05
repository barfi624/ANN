import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# Activation Functions
# -----------------------------
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def cross_entropy(pred, target):
    return -np.mean(np.sum(target * np.log(pred + 1e-9), axis=1))

# -----------------------------
# Network Architecture
# -----------------------------
np.random.seed(0)

input_size = X_train.shape[1]   # 4 features
hidden_size = 10
output_size = y_train.shape[1]  # 3 classes

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

lr = 0.1
epochs = 1000

# -----------------------------
# Training
# -----------------------------
for epoch in range(epochs):

    # 🔹 Forward Propagation
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    y_pred = softmax(z2)

    # 🔹 Loss
    loss = cross_entropy(y_pred, y_train)

    # 🔹 Backpropagation
    d_z2 = y_pred - y_train
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * relu_derivative(z1)
    d_W1 = np.dot(X_train.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # 🔹 Update
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# -----------------------------
# Testing
# -----------------------------
z1 = np.dot(X_test, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
y_pred_test = softmax(z2)

predictions = np.argmax(y_pred_test, axis=1)
actual = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == actual)
print("\nTest Accuracy:", accuracy)