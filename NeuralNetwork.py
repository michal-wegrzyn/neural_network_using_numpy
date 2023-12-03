import numpy as np
import random

class NeuralNetwork:
    def __init__(self, sizes, activation_functions, learning_rate):
        assert len(sizes) == len(activation_functions) + 2
        self.sizes = sizes
        self.activation_functions = activation_functions + [(NeuralNetwork.softmax, None)]
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
        
        d = aa[-1]
        d[test[1]][0] -= 1

        dw[-1] = np.dot(d, aa[-2].transpose())
        db[-1] = d
        for layer in range(2,len(self.sizes)):
            d = np.dot(self.weights[-layer+1].transpose(), d) * self.activation_functions[-layer][1](zz[-layer])
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
    
    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
    
    @staticmethod
    def d_sigmoid(z):
        return NeuralNetwork.sigmoid(z)*(1-NeuralNetwork.sigmoid(z))
    
    sigmoid_pair = (sigmoid, d_sigmoid)

    @staticmethod
    def ReLU(z):
        return np.maximum(0,z)
    
    @staticmethod
    def d_ReLU(a):
        return 1.0 * (a>0)
    
    ReLU_pair = (ReLU, d_ReLU)

    @staticmethod
    def softmax(z):
        exps = np.exp(z - np.max(z))
        if np.nan in exps:
            print('nan', exps)
        return exps / np.sum(exps)