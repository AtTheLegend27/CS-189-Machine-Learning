#Question 4 Part 2: Gradient Descent Class

#imports
import numpy as np

class GradientDescent:
    
    def __init__(self, learn_rate, reg_param, max_iter):
        self.learning_rate = learn_rate
        self.regularization_parameter = reg_param
        self.max_iterations = max_iter
        
        self.weights = None
        self.cost_history = []
        
        
    #from question 3
    def s(self, x):
        sigmoid = 1/(1+np.exp(-x))
        return sigmoid
    
    
    #cost function
    def cost_function(self, X, y, min):
        x_shape = X.shape[0]
        intermed = self.s(np.dot(X, self.weights))
        loss = -np.mean(y * np.log(intermed + min) + (1 - y) * np.log(1 - intermed + min))
        reg = (self.regularization_parameter / (2 * x_shape)) * np.sum(self.weights[1:] ** 2)
        return loss + reg


    #gradient
    def batch_gradient(self, X, y):
        x_shape = X.shape[0]
        intermed = self.s(np.dot(X, self.weights))
        grad = np.dot(X.T, (intermed - y)) / x_shape
        grad[1:] += (self.regularization_parameter / x_shape) * self.weights[1:]
        return grad
    
    
    #fitting
    def fit(self, X, y, min):
        x_shape1 = X.shape[1]
        
        #instantiate weights as 0 to start
        self.weights = np.zeros(x_shape1)
        
        #iterations to fit
        for i in range(self.max_iterations):
            grad = self.batch_gradient(X, y)
            self.weights -= self.learning_rate * grad
            
            cost = self.cost_function(X, y, min)
            self.cost_history.append(cost)
            
            
    #prediction
    def pred(self, X):
        return (self.s(np.dot(X, self.weights)) >= 0.5).astype(int)