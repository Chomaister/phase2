import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils
import math as math


# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)
        print(X.T@X, X.T@y)
        print(self.w)
        
    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
# inherits the predict() function from LeastSquares
class WeightedLeastSquares(LeastSquares): 
    def fit(self,X,y,z):
        self.w = solve(X.T@z@X, X.T@z@y)

    def predict(self, X):
        return X@self.w

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        #a = self.funObj(self.w,X,y)[0]
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        #f = 0.5*np.sum((X@w - y)**2)
        f = 0
        g = 0
        for n in range(0,len(X)):
            f += X[n][0] * w - y[n][0]
            xi = X[n][0]
            yi = y[n][0]
            g += (xi*math.exp(w*xi-yi) - yi*math.exp(yi-w*xi)) / (math.exp(w*xi - yi) + math.exp(yi-w*xi))

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        w0 = [200 for n in range(len(X[0]))]
        return np.add(X@self.w, w0)

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        #self.w = solve(X.T@X, X.T@y)
        a = self.__polyBasis(X)

        self.w = solve(a.T@a, a.T@y)


    def predict(self, X):
        return self.__polyBasis(X)@self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        pb = np.ndarray(shape=(len(X), self.p+1))
        for n in range(len(X)):
            row = pb[n]
            for m in range(0, self.p+1):
                row[m] =  X[n]**m
        return pb