import numpy as np
import random

class NeuralNetwork:
    from ActivationFunctions import ActivationFunctions as AFunc
    from LossFunctions import LossFunctions as LFunc
    
    def __init__(self, sizes, activation_functions, loss_function, learning_rate):
        assert len(sizes) == len(activation_functions) + 1
        self.sizes = sizes
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.biases = [np.random.rand(size, 1) - 0.5 for size in sizes[1:]]
        self.weights = [np.random.rand(sizes[i+1], sizes[i]) - 0.5 for i in range(len(sizes)-1)]

    def forward_propagation(self, inp):
        inp = np.array(inp).reshape(-1,1)
        for b, w, f in zip(self.biases, self.weights, self.activation_functions):
            inp = f[0](np.dot(w,inp)+b)
        return inp

    def train(self, data, epochs, batch_size, show_epoch=None, test_data=None):
        if show_epoch is not None:
            print('Training started')
        data_len = len(data)
        for epoch in range(1,epochs+1):
            random.shuffle(data)
            batches = [data[i:min(i+batch_size,data_len)] for i in range(0, data_len, batch_size)]
            for batch in batches:
                w_sgd, b_sgd = self.stochastic_gradient_descent(batch)
                for i in range(len(self.weights)):
                    self.weights[i] -= w_sgd[i] * self.learning_rate
                for i in range(len(self.biases)):
                    self.biases[i] -= b_sgd[i] * self.learning_rate
            if show_epoch is not None:
                if epoch%show_epoch == 0:
                    print('Epoch', epoch, 'finished')
                    if test_data is not None:
                        print('Accuracy', self.test_accuracy(test_data))
        if show_epoch is not None:
            print('Completed', epochs, 'epochs')
        if test_data is not None:
            print('Accuracy:', self.test_accuracy(test_data))

    def stochastic_gradient_descent(self, batch_of_tests):
        w_sgd = [np.zeros(w.shape) for w in self.weights]
        b_sgd = [np.zeros(b.shape) for b in self.biases]
        for test in batch_of_tests:
            backward_propagation_res = self.backward_propagation(test)
            for i in range(len(b_sgd)):
                b_sgd[i] += backward_propagation_res[1][i]
            for i in range(len(w_sgd)):
                w_sgd[i] += backward_propagation_res[0][i]

        return w_sgd, b_sgd

    def backward_propagation(self, test):
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        a = test[0]
        aa = [a]
        zz = []
        for b, w, f in zip(self.biases, self.weights, self.activation_functions):
            z = np.dot(w,a) + b
            zz.append(z)
            a = f[0](z)
            aa.append(a)
        
        y = test[1]
        
        if type(y) != np.ndarray:
            y = np.array(y)
            if y.shape == ():
                y = self.one_hot(y)

        d = self.loss_function[1](aa[-1], y)
        dz = self.activation_functions[-1][1](zz[-1])
        if dz.shape[1] == 1:
            d = dz*d
        else:
            d = np.dot(dz, d)

        dw[-1] = np.dot(d, aa[-2].transpose())
        db[-1] = d
        for layer in range(2,len(self.sizes)):
            d = np.dot(self.weights[-layer+1].transpose(), d)
            dz = self.activation_functions[-layer][1](zz[-layer])
            if dz.shape[1] == 1:
                d = dz*d
            else:
                d = np.dot(dz, d)
            dw[-layer] = np.dot(d, aa[-layer-1].transpose())
            db[-layer] = d

        return (dw, db)

    def make_prediction(self, data):
        output = self.forward_propagation(data)
        max_arg = np.argmax(output)
        return max_arg

    def test_accuracy(self, tests):
        ok = 0
        for test in tests:
            output = self.make_prediction(test[0])
            if output == test[1]:
                ok += 1
        return ok / len(tests)

    def one_hot(self, ans):
        y = np.zeros((self.sizes[-1], 1))
        y[ans] = 1
        return y
