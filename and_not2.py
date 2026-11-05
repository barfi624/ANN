import numpy as np

# McCulloch-Pitts Neuron
def mp_neuron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    return 1 if weighted_sum >= threshold else 0

# ANDNOT Function: x1 AND (NOT x2)
def and_not(x1, x2):
    weights = [1, -1]   # x1 = excitatory, x2 = inhibitory
    threshold = 1
    inputs = np.array([x1, x2])
    return mp_neuron(inputs, weights, threshold)

# Test all input combinations
print("x1 x2 | Output")
print("--------------")
test_inputs = [(0,0), (0,1), (1,0), (1,1)]

for x1, x2 in test_inputs:
    print(x1, x2, " | ", and_not(x1, x2))