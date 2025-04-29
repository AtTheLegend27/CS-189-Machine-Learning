#Question 6: Isocontours of Normal Distributions

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

#Creating grid plot 10 left, 10 right, 10 up, 10 down and 500 points in all directions
x_space = np.linspace(-10, 10, 500)
y_space = np.linspace(-10, 10, 500)
X,Y = np.meshgrid(x_space, y_space)


#Question 6 Part 1

#Instantiate Part 1 desired inputs
q6p1_u = [1, 1]
q6p1_ep = [[1, 0], [0, 2]]

#Evaluation of the Probability Density Function
position = np.array([Y, X]).T
rand_var = sc.stats.multivariate_normal(q6p1_u, q6p1_ep)
Z_1 = rand_var.pdf(position)

#Plotting Contours
plt.contourf(X, Y, Z_1)
plt.colorbar()
plt.title("Question 6 Part 1 Isocontours")
plt.show()



#Question 6 Part 2

#Instantiate Part 2 desired inputs
q6p2_u = [-1, 2]
q6p2_ep = [[2, 1], [1, 4]]

#Evaluation of the Probability Density Function
position_2 = np.array([Y, X]).T
rand_var_2 = sc.stats.multivariate_normal(q6p2_u, q6p2_ep)
Z_2 = rand_var_2.pdf(position_2)

#Plotting Contours
plt.contourf(X, Y, Z_2)
plt.colorbar()
plt.title("Question 6 Part 2 Isocontours")
plt.show()


#Question 6 Part 3

#Instantiate Part 3 desired inputs
q6p3_u = [0, 2]
q6p3_u2 = [2, 0]
q6p3_ep = [[2, 1], [1, 1]]

#Evaluation of the Probability Density Function
position_3 = np.array([Y, X]).T
rand_var_3 = sc.stats.multivariate_normal(q6p3_u, q6p3_ep)
rand_var_3_2 = sc.stats.multivariate_normal(q6p3_u2, q6p3_ep)
Z_3 = rand_var_3.pdf(position_3) - rand_var_3_2.pdf(position_3)

#Plotting Contours
plt.contourf(X, Y, Z_3)
plt.colorbar()
plt.title("Question 6 Part 3 Isocontours")
plt.show()


#Question 6 Part 4

#Instantiate Part 4 desired inputs
q6p4_u = [0, 2]
q6p4_u2 = [2, 0]
q6p4_ep = [[2, 1], [1, 1]]
q6p4_ep2 = [[2, 1], [1, 4]]

#Evaluation of the Probability Density Function
position_4 = np.array([Y, X]).T
rand_var_4 = sc.stats.multivariate_normal(q6p4_u, q6p4_ep)
rand_var_4_2 = sc.stats.multivariate_normal(q6p4_u2, q6p4_ep2)
Z_4 = rand_var_4.pdf(position_4) - rand_var_4_2.pdf(position_4)

#Plotting Contours
plt.contourf(X, Y, Z_4)
plt.colorbar()
plt.title("Question 6 Part 4 Isocontours")
plt.show()


#Question 6 Part 5

#Instantiate Part 5 desired inputs
q6p5_u = [1, 1]
q6p5_u2 = [-1, -1]
q6p5_ep = [[2, 0], [0, 1]]
q6p5_ep2 = [[2, 1], [1, 2]]

#Evaluation of the Probability Density Function
position_5 = np.array([Y, X]).T
rand_var_5 = sc.stats.multivariate_normal(q6p5_u, q6p5_ep)
rand_var_5_2 = sc.stats.multivariate_normal(q6p5_u2, q6p5_ep2)
Z_5 = rand_var_5.pdf(position_5) - rand_var_5_2.pdf(position_5)

#Plotting Contours
plt.contourf(X, Y, Z_5)
plt.colorbar()
plt.title("Question 6 Part 5 Isocontours")
plt.show()