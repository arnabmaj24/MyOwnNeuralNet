#goal is to make my own ML library
import numpy as np
import time

def mse(yt, yp):
    return np.mean(np.power(yt - yp, 2))

def dmse(yt, yp):
    return 2 * (yp - yt) / np.size(yt)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_step(self, input):
        pass

    def backward_step(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, inpsize, outsize):

        #weight matrix and bias vector

        self.weights = np.random.randn(outsize, inpsize)
        self.bias = np.random.randn(outsize, 1)

    def forward_step(self, input):
        self.input = input
        # Y = W * X + B
        return np.dot(self.weights, self.input) + self.bias

    def backward_step(self, output_gradient, learning_rate):
        #dE/dW = dE/DY * X^T
        weights_gradient = np.dot(output_gradient, self.input.T)
        # W = W - dE/dW * alpha
        self.weights -= weights_gradient * learning_rate
        # B = B - dE/DY * alpha
        self.bias -= output_gradient * learning_rate
        # dE/dX = W^T * dE/dY
        return np.dot(self.weights.T, output_gradient)

class Activiation(Layer):
    def __init__(self, activiation, dA):
        self.activation = activiation
        self.dA = dA

    def forward_step(self, input):
        self.input = input
        return self.activation(self.input)

    def backward_step(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.dA(self.input))

class Tanh(Activiation):
    def __init__(self):
        # tanh = lambda x: np.tanh(x)
        # dtanh = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(self.tanh, self.dtanh)

    def tanh(self,x):
        return np.tanh(x)

    def dtanh(self,x):
        return 1 - self.tanh(x)**2

class Sigmoid(Activiation):
    def __init__(self):
        super().__init__(self.sigmoid, self.dsigmoid)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

class ReLU(Activiation):
    def __init__(self):
        super().__init__(self.relu, self.drelu)

    def relu(self, x):
        return np.maximum(0, x)

    def drelu(self, x):
        return np.where(x > 0, 1, 0)

class Softmax(Layer):
    def forward_step(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward_step(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward_step(output)
    return output

def train(network, loss, dloss, X, Y, epochs = 1000, LR = 0.01, printout = True):
    start = time.time()
    for e in range(epochs):
        error = 0
        for x, y in zip(X, Y):

            #forward step
            output = predict(network, x)

            #error calc
            error += loss(y, output)

            #backward step
            gradient = dloss(y, output)
            for layer in reversed(network):
                gradient = layer.backward_step(gradient, LR)

        error /= len(X)
        if printout:
            print(f"{e + 1}/{epochs}, error={error}, time passed={round(time.time()-start, 2)}s" )