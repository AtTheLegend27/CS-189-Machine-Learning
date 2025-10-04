#Question 4 Part 4: Stochastic Gradient Descent Class

#imports
import numpy as np

class SGradientDescent:
    
    def __init__(self, delta, learn_rate, reg_param, max_iter):
        self.delta = delta
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
    def stochastic_gradient(self, X_i, y_i):
        intermed = self.s(np.dot(X_i, self.weights))  
        grad = X_i.T * (intermed - y_i)  
        grad[1:] += self.regularization_parameter * self.weights[1:] 
        return grad

    #fitting
    def fit(self, X, y, min):
        x_shape1 = X.shape[1]
        self.weights = np.zeros(x_shape1) 

        for t in range(1, self.max_iterations + 1):
            #important: shuffle again before fitting to avoid bias
            idxs = np.random.permutation(len(y))
            X, y = X[idxs], y[idxs]

            for j in range(len(y)):  
                grad = self.stochastic_gradient(X[j], y[j])
                self.weights -= self.learning_rate * grad

            cost = self.cost_function(X, y, min)
            self.cost_history.append(cost)
            
            
    #decay fitting
    def d_fit(self, X, y, min):
        x_shape1 = X.shape[1]
        self.weights = np.zeros(x_shape1) 

        for t in range(1, self.max_iterations + 1):
            #important: shuffle again before fitting to avoid bias
            idxs = np.random.permutation(len(y))
            X, y = X[idxs], y[idxs]

            for j in range(len(y)):  
                grad = self.stochastic_gradient(X[j], y[j])
                learning_rate = self.delta / t
                self.weights -= learning_rate * grad

            cost = self.cost_function(X, y, min)
            self.cost_history.append(cost)

    #prediction
    def pred(self, X):
        return (self.s(np.dot(X, self.weights)) >= 0.5).astype(int)
