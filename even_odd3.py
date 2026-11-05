import numpy as np

# Input number
j = int(input("Enter a Number (0-9): "))

# Step activation function
def step(x):
    return 1 if x >= 0 else 0

# Training data (6-bit binary representation of 0–9)
training_data = [
    ([0,0,0,0,0,0], 1),  # 0 even
    ([0,0,0,0,0,1], 0),  # 1 odd
    ([0,0,0,0,1,0], 1),  # 2 even
    ([0,0,0,0,1,1], 0),  # 3 odd
    ([0,0,0,1,0,0], 1),  # 4 even
    ([0,0,0,1,0,1], 0),  # 5 odd
    ([0,0,0,1,1,0], 1),  # 6 even
    ([0,0,0,1,1,1], 0),  # 7 odd
    ([0,0,1,0,0,0], 1),  # 8 even
    ([0,0,1,0,0,1], 0),  # 9 odd
]

# Initialize weights and bias
weights = np.zeros(6)
bias = 0
lr = 0.1
epochs = 20

# Training (Perceptron Learning Rule)
for _ in range(epochs):
    for x, y in training_data:
        x = np.array(x)
        z = np.dot(x, weights) + bias
        output = step(z)

        error = y - output

        weights += lr * error * x
        bias += lr * error

# Testing input
binary_input = np.array([int(bit) for bit in format(j, '06b')])
z = np.dot(binary_input, weights) + bias
prediction = step(z)

parity = "even" if prediction == 1 else "odd"

print("Binary:", binary_input)
print(j, "is", parity)