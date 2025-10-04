#Question 6 Part 1

#imports
import numpy as np
import matplotlib.pyplot as plt

def l_p_norm(X, Y, p):
    sum = (np.abs(X)**p + np.abs(Y)**p)**(1/p)
    return sum

X_grid = np.linspace(-5, 5, 1000)
Y_grid = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(X_grid, Y_grid)

#Part a: l_0.5 norm
z_dim_a = l_p_norm(X, Y, 0.5)
plt.contour(X, Y, z_dim_a)
plt.title("Lp normalized 0.5 Contour Plot")
plt.xlabel("X_Dim")
plt.ylabel("Y_Dim")
plt.show()

#Part b: l_1 norm
z_dim_b = l_p_norm(X, Y, 1)
plt.contour(X, Y, z_dim_b)
plt.title("Lp normalized 1 Contour Plot")
plt.xlabel("X_Dim")
plt.ylabel("Y_Dim")
plt.show()

#Part c: l_2 norm
z_dim_c = l_p_norm(X, Y, 2)
plt.contour(X, Y, z_dim_c)
plt.title("Lp normalized 2 Contour Plot")
plt.xlabel("X_Dim")
plt.ylabel("Y_Dim")
plt.show()