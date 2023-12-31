import matplotlib.pyplot
import time
import numpy as np
import pandas as pd

from nn import Dense
from nn import Tanh, Sigmoid, ReLU
from nn import mse, dmse, train, predict
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils



dataTR = pd.read_csv("/mnist/mnist_train.csv")
dataTE = pd.read_csv("/mnist/mnist_test.csv")

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)


network = [
    Dense(28*28, 40),
    Tanh(),
    Dense(40, 16),
    Tanh(),
    Dense(16, 10),
    Sigmoid()
]

train(network, mse, dmse, x_train, y_train, epochs=100, LR=0.1)

accuracy = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    if np.argmax(output) == np.argmax(y):
        accuracy +=1
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y), )

print(f"correct: {accuracy/len(x_test)*100}%")