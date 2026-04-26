import numpy as np

e = 2.718281828459045

def sigmoid(x):
    return 1 / (1 + e ** (-x))

def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


#A1: Neuron class without numpy
class Neuron: 
    def __init__(self, weights, bias, activation=None):
        self.weights = weights # list of weights
        self.bias = bias # single bias value
        self.activation = activation # activation functionc or none
    
    def weighted_sum(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match number of weights")

        z = 0.0
        for i in range(len(inputs)):
            z += inputs[i] * self.weights[i]
        z += self.bias
        return z

    def forward(self, inputs):
        z = self.weighted_sum(inputs)
        if self.activation is None:
            return z
        return self.activation(z)

#A2: Neuron class with numpy
class NeuronNumpy:
    def __init__(self, weights, bias, activation=None):
        self.weights = np.array(weights, dtype=float)
        self.bias = float(bias)
        self.activation = activation

    def weighted_sum(self, inputs):
        x = np.array(inputs, dtype=float)
        if x.shape != self.weights.shape:
            raise ValueError("inputs och weights måste ha samma form")
        z = self.weights @ x + self.bias
        return z

    def forward(self, inputs):
        z = self.weighted_sum(inputs)
        if self.activation is None:
            return z
        return self.activation(z)

#B: Dense layer class with numpy
class DenseLayer:
    def __init__(self, weights, bias, activation=None):
        # weights: shape (n_neurons, n_inputs)
        # bias:    shape (n_neurons,)
        self.W = np.array(weights, dtype=float)
        self.b = np.array(bias, dtype=float)
        self.activation = activation
        

    def forward(self, x):
        x = np.array(x, dtype=float)  # shape (n_inputs,)
        z = self.W @ x + self.b  # shape (n_neurons,)
        if self.activation is None:
            return z
        return self.activation(z)

if __name__ == "__main__":
    A = [1.2, 3.4, 5.6]
    W = [0.5, 1.8, -1.3]
    B = 0.1

    #A1: Neuron class without numpy
    n = Neuron(weights=W, bias=B, activation=sigmoid)
    z = n.weighted_sum(A)
    y = n.forward(A)

    print("Weighted sum:", z)
    print("After activation:", y)

    #A2: Neuron class with numpy
    n2 = NeuronNumpy(weights=W, bias=B, activation=sigmoid)

    print("Weighted sum with numpy:", n2.weighted_sum(A))
    print("After activation with numpy:", n2.forward(A))

#B: Dense layer class with numpy

x = [2.0, -1.0, 0.5, 3.0]  # 4 inputs
# 3 neuroner i lagret, så weights är (3 x 4) och bias är (3,)
W_layer = [
    [0.2,  -0.1, 0.4,  1.0],   # neuron 1
    [-1.2,  0.3, 0.0, -0.7],   # neuron 2
    [0.8,   0.8, 0.8,  0.8],   # neuron 3
]
b_layer = [0.1, -0.2, 0.0]

layer = DenseLayer(W_layer, b_layer, activation=sigmoid_np)
y = layer.forward(x)

print("Output from dense layer:", y)

