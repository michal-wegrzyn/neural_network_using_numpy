from NeuralNetwork import NeuralNetwork as NN
from DatasetAugmentation import DatasetAugmentation as augment

import _pickle
import gzip

nn = NN([784, 16, 16, 10], [NN.AFunc.SELU_pair, NN.AFunc.PReLU_pair(0.01), NN.AFunc.softmax_pair], NN.LFunc.cross_entropy_pair, 0.001)

with gzip.open('mnist.pkl.gz', 'rb') as file:
    (x_train, y_train), (x_test, y_test) = _pickle.load(file)

x_train = x_train / 255
x_test = x_test / 255

train_data = [list(data) for data in zip(x_train, y_train)]
train_data = augment.shift(train_data, [[0,0],[1,0],[-1,0],[0,1],[0,-1]], 0.0, (28,28), (784,1))
print(len(train_data), 'training examples')
test_data = [(x_test[i].reshape(784, 1), y_test[i]) for i in range(y_test.shape[0])]

nn.train(train_data, 4, 64, 1, test_data)