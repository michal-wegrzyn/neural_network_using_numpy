import numpy as np

class ActivationFunctions:
    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    @staticmethod
    def d_sigmoid(z):
        return ActivationFunctions.sigmoid(z)*(1-ActivationFunctions.sigmoid(z))

    sigmoid_pair = (sigmoid, d_sigmoid)

    @staticmethod
    def ReLU(z):
        return np.maximum(0,z)

    @staticmethod
    def d_ReLU(z):
        return 1.0 * (z>0)

    ReLU_pair = (ReLU, d_ReLU)

    @staticmethod
    def PReLU(alpha):
        return lambda z: np.where(z>0, z, z*alpha)
    
    @staticmethod
    def d_PReLU(alpha):
        return lambda z: np.where(z>0, 1, alpha)
    
    @staticmethod
    def PReLU_pair(alpha):
        return (ActivationFunctions.PReLU(alpha), ActivationFunctions.d_PReLU(alpha))
    
    @staticmethod
    def ELU(alpha):
        return lambda z: np.where(z>0, z, alpha*(np.exp(z)-1))
    
    @staticmethod
    def d_ELU(alpha):
        return lambda z: np.where(z>0, 1, alpha*np.exp(z))
    
    @staticmethod
    def ELU_pair(alpha):
        return (ActivationFunctions.ELU(alpha), ActivationFunctions.d_ELU(alpha))
    
    @staticmethod
    def SELU(z):
        return 1.0507 * ActivationFunctions.ELU(1.67326)(z)

    @staticmethod
    def d_SELU(z):
        return 1.0507 * ActivationFunctions.d_ELU(1.67326)(z)
    
    SELU_pair = (SELU, d_SELU)

    @staticmethod
    def softmax(z):
        exps = np.exp(z - np.max(z))
        return exps / np.sum(exps)

    @staticmethod
    def d_softmax(z):
        a = ActivationFunctions.softmax(z)
        return np.diagflat(a) - np.dot(a, a.transpose())

    softmax_pair = (softmax, d_softmax)
