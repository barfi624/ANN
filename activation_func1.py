import numpy as np
import matplotlib.pyplot as plt

# Define input range
x = np.linspace(-10, 10, 400)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (2 / (1 + np.exp(-2*x))) - 1

def relu(x):
    return np.maximum(0, x)



def softplus(x):
    return np.log(1 + np.exp(x))

# Plotting
plt.figure(figsize=(10, 8))

#plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
'''plt.plot(x, relu(x), label="ReLU")
plt.plot(x, softplus(x), label="Softplus")'''

# Formatting
plt.title("Common Activation Functions in Neural Networks")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)

plt.show()