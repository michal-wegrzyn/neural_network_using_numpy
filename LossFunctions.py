import numpy as np

class LossFunctions:
    @staticmethod
    def cross_entropy(a, y):
        return - np.sum(y*np.log(a))

    @staticmethod
    def d_cross_entropy(a, y):
        return -y/a

    cross_entropy_pair = (cross_entropy, d_cross_entropy)

    @staticmethod
    def quadratic(a,y):
        return np.sum(np.square(a-y))

    @staticmethod
    def d_quadratic(a,y):
        return 2*(a-y)

    quadratic_pair = (quadratic, d_quadratic)

    @staticmethod
    def absolute_error(a,y):
        return np.sum(np.abs(a-y))
    
    @staticmethod
    def d_absolute_error(a,y):
        return np.where(a>y, 1, -1)
    
    absolute_error_pair = (absolute_error, d_absolute_error)