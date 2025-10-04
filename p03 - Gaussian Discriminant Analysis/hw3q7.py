#Question 7: EigenVectors of Gaussian Covariance Matrix

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(7)


#Drawing Random Samples from a Normal (Gaussian) Dist
X1 = np.random.normal(loc=3, scale=3, size=100)
norm_44 = np.random.normal(loc=4, scale=2, size=100)
X2 = 0.5 * X1 + norm_44

sample = np.vstack((X1, X2)).T


#Question 7 Part 1
sample_mean = np.mean(sample, axis = 0)
print("Sample Mean in R squared is: ", sample_mean)


#Question 7 Part 2
sample_covariance = np.cov(sample, rowvar = False)
print("Sample Covariance in R squared is: ", sample_covariance)


#Question 7 Part 3
eigen_values, eigen_vectors = np.linalg.eig(sample_covariance)
print("The Eigenvalues for this sample covariance: ", eigen_values)
print("The Eigenvectors for this sample covariance: ", eigen_vectors)



#Questions 7 Part 4/5 Plotting Set Up
left_bound = -15
right_bound = 15



#Question 7 Part 4
vector_1_mean = [sample_mean[0], sample_mean[0]]
vector_2_mean = [sample_mean[1], sample_mean[1]]
eig_vector_and_val_1 = [eigen_vectors[0][0] * eigen_values[0], eigen_vectors[0][1] * eigen_values[1]]
eig_vector_and_val_2 = [eigen_vectors[1][0] * eigen_values[0], eigen_vectors[1][1] * eigen_values[1]]


plt.figure(figsize= (7, 7))
plt.scatter(sample[:, 0], sample[:, 1])
plt.xlim(left_bound, right_bound)
plt.ylim(left_bound, right_bound)
plt.quiver(vector_1_mean, vector_2_mean, eig_vector_and_val_1, eig_vector_and_val_2, angles = "xy", scale_units = "xy", scale = 1)

plt.title("Question 7 Part 4: Sample n = 100 points and Covariance EigenVectors")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


#Question 7 Part 5
rotation = np.dot(eigen_vectors.T, (sample-sample_mean).T).T

plt.figure(figsize= (7, 7))

plt.scatter(rotation[:, 0], rotation[:, 1])
plt.xlim(left_bound, right_bound)
plt.ylim(left_bound, right_bound)

plt.title("Question 7 Part 5: Sample n = 100 points after Rotation")
plt.xlabel("Rotated X1")
plt.ylabel("Rotated X2")
plt.show()