from nn import Dense
from nn import Tanh, Sigmoid, ReLU
from nn import mse, dmse
import numpy as np

X = np.reshape([[0,0],[1,0],[0,1],[0,0]], (4,2,1))
Y = np.reshape([[0],[1],[1], [0]], (4,1,1))

network = [
    Dense(2,3),
    ReLU(),
    Dense(3,1),
    ReLU()
]

epochs = 10000
learnrate = 0.1

for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        out = x
        for layer in network:
            out = layer.forward_step(out)

        error += mse(y, out)

        grad = dmse(y, out)
        for layer in reversed(network):
            grad = layer.backward_step(grad, learnrate)

        error /= len(X)
        print('%d/%d, error = %f' % (e+1, epochs, error))