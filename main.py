from NeuralNetwork import NeuralNetwork as NN
import numpy as np
import _pickle
import gzip

nn = NN([784, 16, 16, 10], [NN.ReLU_pair, NN.ReLU_pair], 0.001)

with gzip.open('mnist.pkl.gz', 'rb') as file:
    (x_train, y_train), (x_test, y_test) = _pickle.load(file)

x_train = x_train / 255
x_test = x_test / 255

x_train = [np.pad(x_train[i].reshape(28,28), 1, 'constant', constant_values=0) for i in range(y_train.shape[0])]

train_data = [(x_train[i][1:-1, 1:-1].reshape(784, 1), y_train[i]) for i in range(y_train.shape[0])]
train_data += [(x_train[i][1:-1, :-2].reshape(784, 1), y_train[i]) for i in range(y_train.shape[0])]
train_data += [(x_train[i][1:-1, 2:].reshape(784, 1), y_train[i]) for i in range(y_train.shape[0])]
train_data += [(x_train[i][:-2, 1:-1].reshape(784, 1), y_train[i]) for i in range(y_train.shape[0])]
train_data += [(x_train[i][2:, 1:-1].reshape(784, 1), y_train[i]) for i in range(y_train.shape[0])]
print(len(train_data), 'training examples')
test_data = [(x_test[i].reshape(784, 1), y_test[i]) for i in range(y_test.shape[0])]

nn.train(train_data, 4, 64, 1, test_data)