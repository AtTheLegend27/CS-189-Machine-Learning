#Homework 4 Question 3 Part 4

#imports
import numpy as np

#set up

#X Design Matrix with imaginary consideration 3rd with value 1 called bias term
X = np.array([
    [0.2, 3.1, 1],
    [1.0, 3.0, 1],
    [-0.2, 1.2, 1],
    [1.0, 1.1, 1],
])

y = np.array([1, 1, 0, 0])

w0 = np.array([-1, 1, 0])

def s(x):
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid

def calculate_si(X, wi):
    si = s(np.dot(X, wi))
    return si

def calculate_wi(X, si, wi, y):
    middle_term = np.diag(si *(1 - si))
    total_term_A = X.T.dot(middle_term).dot(X)
    
    total_term_A_inv = np.linalg.inv(total_term_A)
    
    first_gradient_cost = X.T.dot(y-si)
    right_term = np.dot(total_term_A_inv, first_gradient_cost)
    wi_wt = wi + right_term
    
    return wi_wt
    
    

#Part A
s0 = calculate_si(X, w0)
print("The value(s) of s0 is:", s0)
#The value(s) of s0 is: [0.94784644 0.88079708 0.80218389 0.52497919]


#Part B
w1 = calculate_wi(X, s0, w0, y)
print("The value(w) of w1 is:", w1)
#The value(w) of w1 is: [ 1.32465198  3.04991697 -6.82910388]

#Part C
s1 = calculate_si(X, w1)
print("The value(s) of s1 is:", s1)
#The value(s) of s1 is: [0.94737826 0.97455097 0.03124556 0.10437391]

#Part B
w2 = calculate_wi(X, s1, w1, y)
print("The value(w) of w2 is:", w2)
#The value(w) of w2 is: [ 1.36602464  4.15753654 -9.19961627]