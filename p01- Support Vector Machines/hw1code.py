##HW 1 Code

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd 

random.seed(189)
np.random.seed(189)

# #Question 2E Plot Code

# Parameters for w & alpha + load-toy data specific to training data
w = np.array([-0.4528, -0.5190])
alpha = 0.1471
data = np.load("/Users/dankim/Downloads/Cal Stuff/Cal Coursework F2020-Sp2025/Cal Berkeley Spring 2025/CS 189/Homework/hw1/data/toy-data.npz")
training_data = data["training_data"]
decision_values = np.dot(training_data, w) + alpha
labels = np.sign(decision_values)  


# Support Vectors
pre_support_vector = np.isclose(np.abs(decision_values), 1, atol=0.05)
support_vectors = training_data[pre_support_vector]


# Plot the data points 
def plot_data_points(training_data, labels): 
    plt.scatter(training_data[:, 0], training_data[:, 1], c=labels) 

# Plot the decision boundary 
def plot_decision_boundary(w, b): 
    x = np.linspace(-5, 5, 100) 
    y =-(w[0] * x + b) / w[1] 
    plt.plot(x, y, 'k')

#Plot the margins 
def plot_margins(w, b):
    x = np.linspace(-5, 5, 100)
    y1 = -(w[0] * x + b + 1) / w[1]  
    y2 = -(w[0] * x + b - 1) / w[1] 
    plt.plot(x, y1, 'r--', label="Margin +1")
    plt.plot(x, y2, 'g--', label="Margin -1")
    
# Plot complete Graph
plot_data_points(training_data, labels)
plot_decision_boundary(w, alpha)
plot_margins(w, alpha)

#Highlight support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, edgecolors='k', linewidths=2, label="Support Vectors")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Q2E: Linear SVM Decision Boundary with Margins and Support Vectors")
plt.show()



# Question 3 code: Data Partitioning and Evaluation Metrics

# Question 3A: Data Partitioning for MNIST and spam
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
    
    
# Question 3B: Evaluation Metric
def evaluation_metric(y_vals, y_predictions):
    return np.mean(y_vals == y_predictions)


# Question 4: Support Vector Machines: Coding
def plot_classification_accuracy(trained_size, train_accuracy, validation_accuracy, data_title):
    plt.title(data_title)
    plt.plot(trained_size, train_accuracy, label="Training Set")
    plt.plot(trained_size, validation_accuracy, label="Validation Set")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Classification Accuracy")
    plt.legend()
    plt.show()
     
def SVM_training(train_data, train_labels, validation_data, validation_labels):
    svm_model = SVC(kernel="linear")
    svm_model.fit(train_data, train_labels)
    
    train_predictions = svm_model.predict(train_data)
    validation_predictions = svm_model.predict(validation_data)
    
    train_accuracy = evaluation_metric(train_labels, train_predictions)
    validation_accuracy = evaluation_metric(validation_labels, validation_predictions)
    
    return train_accuracy, validation_accuracy

def individual_SVM_training(total_list, train_data, train_label, validation_data, validation_label):
    list1 = []
    list2 = []
    for size in total_list:
        train_acc, validation_acc = SVM_training(train_data[:size], train_label[:size], validation_data, validation_label)
        list1.append(train_acc)
        list2.append(validation_acc)
    return list1, list2
    

#Checking Question 4 and 3 A/B
ms_training_size = [100, 200, 500, 1000, 2000, 5000, 10000]


#spam Plot
spam_loaded_data = np.load("/Users/dankim/Downloads/Cal Stuff/Cal Coursework F2020-Sp2025/Cal Berkeley Spring 2025/CS 189/Homework/hw1/data/spam-data.npz")
s_training_data = spam_loaded_data["training_data"]
s_training_labels = spam_loaded_data["training_labels"]
s_num_vals = 0.2

mss_training_size = [100, 200, 500, 1000, 2000, s_training_data.shape[0]]

sM_train_data, sM_train_labels, sM_validation_data, sM_validation_labels = data_partitioning(s_training_data, s_training_labels, s_num_vals)

#Train SVC Model for SVM
s_train_plt_data, s_validation_plt_data = individual_SVM_training(mss_training_size, sM_train_data, sM_train_labels, sM_validation_data, sM_validation_labels)

#Plot Spam
spam_plot = plot_classification_accuracy(mss_training_size, s_train_plt_data, s_validation_plt_data, "Spam Classification Accuracy Results")
spam_plot



# #MNIST Plot
MNIST_loaded_data = np.load("/Users/dankim/Downloads/Cal Stuff/Cal Coursework F2020-Sp2025/Cal Berkeley Spring 2025/CS 189/Homework/hw1/data/mnist-data.npz")
m_training_data = MNIST_loaded_data["training_data"]
m_training_data = m_training_data.reshape(m_training_data.shape[0], -1)
m_training_labels = MNIST_loaded_data["training_labels"]
m_num_vals = 10000

pM_train_data, pM_train_labels, pM_validation_data, pM_validation_labels = data_partitioning(m_training_data, m_training_labels, m_num_vals)

#Train SVC Model for SVM
p_train_plt_data, p_validation_plt_data = individual_SVM_training(ms_training_size, pM_train_data, pM_train_labels, pM_validation_data, pM_validation_labels)

MNIST_plot = plot_classification_accuracy(ms_training_size, p_train_plt_data, p_validation_plt_data, "MNIST Classification Accuracy Results")
MNIST_plot


# Question 5: Hyperparameter Tuning

c_values = [1e-08, 5e-08, 1e-07, 5e-07, 1e-06, 5e-06, 1e-05, 5e-04]

def plot_c_values_validation(c_values, validation_accuracy):
    plt.title("C Values influence on Validation Accuracy")
    plt.plot(c_values, validation_accuracy, label="Validation Set")
    plt.xlabel("C Values")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
    
def c_val_model(x, y, c):
    svm_model = SVC(kernel="linear", C = c)
    svm_model.fit(x, y)
    return svm_model

def c_val_test(train_data, train_labels, validation_data, validation_labels, c_vals):
    val_acc = []
    res = []
    for C in c_vals:
        model = c_val_model(train_data[:10000], train_labels[:10000], C)
        predict = model.predict(validation_data)
        validation_acc = evaluation_metric(predict, validation_labels)
        val_acc.append(validation_acc)
        res.append([C, validation_acc])
    return val_acc, res

m_validation_accuracy, res = c_val_test(pM_train_data, pM_train_labels, pM_validation_data, pM_validation_labels, c_values)
plot_c_values_validation(c_values, m_validation_accuracy)
print(res)


#Question 6: K-Fold Cross-Validation

c_values_spam = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]

def xy_start_end(k, size_val, random_index, train_data, train_labels, total_num):
    x_vals = []
    y_vals = []
    for i in range(k):
        start = i * size_val
        if i == k - 1:
           end = total_num
        else:
            end = (i+1) *  size_val
        x_vals.append(train_data[random_index[start:end]])
        y_vals.append(train_labels[random_index[start:end]])
    return x_vals, y_vals

def k_fold_cross_validation(k, train_data, train_labels, c_vals):
    total_num = len(train_data)
    random_index = np.random.permutation(total_num)
    size_val = total_num // k
    x_vals, y_vals = xy_start_end(k, size_val, random_index, train_data, train_labels, total_num)
     
    res = []
    vali_vals = []
    for C in c_vals:
        validation_values = []
        for i in range(k):
            x_val = x_vals[i]
            y_val = y_vals[i]
            x_train = np.concatenate(x_vals[:i] + x_vals[i+1:], axis = 0)
            y_train = np.concatenate(y_vals[:i] + y_vals[i+1:], axis = 0)
            k_fold_model = c_val_model(x_train[:10000], y_train[:10000], C)
            predict = k_fold_model.predict(x_val)
            validation_values.append(evaluation_metric(predict, y_val))
        res.append([C, np.mean(validation_values)])
        vali_vals = validation_values
    return res, vali_vals
    
k_res, k_validation_accuracy = k_fold_cross_validation(5, sM_train_data, sM_train_labels, c_values_spam)
print(k_res)



# Question 7: Kaggle:
s_test_data = spam_loaded_data["test_data"]
s_test_data = s_test_data.reshape(s_test_data.shape[0], -1)
m_test_data = MNIST_loaded_data["test_data"]
m_test_data = m_test_data.reshape(m_test_data.shape[0], -1)

def results_to_csv(y_test, file_name): 
    y_test = y_test.astype(int) 
    df = pd.DataFrame({"Category": y_test}) 
    df.index += 1 
    df.to_csv(file_name, index_label="Id")
    print(f"CSV saved as {file_name}")

def testing_Kaggle(train_data, train_labels, test_data, c_val, file_name):
    model = c_val_model(train_data, train_labels, c_val)
    test_label = model.predict(test_data)
    results_to_csv(test_label, file_name)

m_question7 = testing_Kaggle(m_training_data, m_training_labels, m_test_data, 5e-07, "MNIST_Test_Labels.csv")
s_question7 = testing_Kaggle(s_training_data, s_training_labels, s_test_data, 10, "spam_Test_Labels.csv")


    

