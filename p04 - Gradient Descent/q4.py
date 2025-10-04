#Homework 4 Question 4

#imports
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

from q4GradDesc import GradientDescent
from q4StoGradDesc import SGradientDescent


#implement random seed
np.random.seed(7)
min_num = 1e-9


#Kaggle CSV function from HW 1
def results_to_csv(y_test, file_name): 
    y_test = y_test.astype(int) 
    df = pd.DataFrame({"Category": y_test}) 
    df.index += 1 
    df.to_csv(file_name, index_label="ID")
    

#Using Data Partitioning Function from Homework #1
def data_partitioning(data, labels, num_vals):
    #Randomness Generator
    num_total = len(data)
    
    #For spam data partioning since its a decimal 0.2 instead of a value like 10k
    if num_vals < 1:
        num_vals = int(num_total * num_vals)
        
    random_index = np.random.permutation(num_total)
    
    #Training Indeces
    train_indices = random_index[num_vals:]
    val_indices = random_index[:num_vals]
    
    #Training Data/Labels
    train_data = data[train_indices]
    train_labels = labels[train_indices]
    
    #Validation Data/Labels
    validation_data = data[val_indices]
    validation_labels = labels[val_indices]
    
    #Return Training Data/Labels & Validation Data/Labels
    return train_data, train_labels, validation_data, validation_labels

    
#load datamat for train/validate/test
dataset = sc.io.loadmat("data.mat")

X_feat = dataset['X']
y_labels = dataset['y'].flatten()

#Normalize 
X_avg = np.average(X_feat, axis=0)
X_std = np.std(X_feat, axis=0)
X_norm = (X_feat - X_avg) / (X_std + min_num)

#bias term
X_norm_bias = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])

#kaggle testing
X_test = dataset['X_test']
X_test_norm = (X_test - X_avg) / (X_std + min_num)
X_test_norm_bias = np.hstack([np.ones((X_test_norm.shape[0], 1)), X_test_norm])


#data partitioned into training/validation
X_train_data, y_train_labels, X_validation_data, y_validation_labels = data_partitioning(X_norm_bias, y_labels, 0.8)
    
    
#Question 4 Part 2: Batch Gradient Descent
grad_desc_model = GradientDescent(learn_rate=0.1, reg_param=0.1, max_iter=10000)

#fitting
grad_desc_model.fit(X_train_data, y_train_labels, min_num)

#predictions
predictions = grad_desc_model.pred(X_validation_data)

#calculate accuracy
accuracy = np.mean(predictions == y_validation_labels)

print("Validation Accuracy = ", accuracy)
#Validation Accuracy =  0.994

#plotting cost and grad desc
plt.plot(grad_desc_model.cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("Batch Gradient Descent Convergence")
plt.show()


#Kaggle for using Batch Gradient Descent
grad_desc_model_total = GradientDescent(learn_rate=0.1, reg_param=0.1, max_iter=10000)
grad_desc_model_total.fit(X_norm_bias, y_labels, min_num)
predictions_total = grad_desc_model_total.pred(X_test_norm_bias)

results_to_csv(predictions_total, "Kaggle_Submission_Grad_Desc_Batch.csv")


#Question 4 Part 4: Stochastic Gradient Descent
s_grad_desc_model = SGradientDescent(delta = None, learn_rate=1e-4, reg_param=0.1, max_iter=10000)

#fitting
s_grad_desc_model.fit(X_train_data, y_train_labels, min_num)

#predictions
s_predictions = s_grad_desc_model.pred(X_validation_data)

#calculate accuracy
s_accuracy = np.mean(s_predictions == y_validation_labels)

print("Validation Accuracy = ", s_accuracy)
#Validation Accuracy =  0.97875

#plotting cost and grad desc
plt.plot(s_grad_desc_model.cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("Stochastic Gradient Descent Convergence")
plt.show()


#Question 5: Decay
s_grad_desc_decay = SGradientDescent(delta=0.001, learn_rate=None, reg_param=0.1, max_iter=10000)
s_grad_desc_decay.d_fit(X_train_data, y_train_labels, min_num)
s_predictions_decay = s_grad_desc_decay.pred(X_validation_data)

s_accuracy_decay = np.mean(s_predictions_decay == y_validation_labels)
print("Validation Accuracy with Learning Rate Decay =", s_accuracy_decay)
#Validation Accuracy with Learning Rate Decay = 0.99025

# Plot the cost function
plt.plot(s_grad_desc_decay.cost_history, label = "SGD Decay")
plt.plot(s_grad_desc_model.cost_history, label = "SGD Fixed")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("SGD Decay vs Fixed Learning Rate")
plt.show()
