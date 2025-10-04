
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm, metrics
import random

random.seed(189)
np.random.seed(189)

# Question 3 code: Data Partitioning and Evaluation Metrics

# Question 3A: Data Partitioning for MNIST and spam
def data_partitioning(data, labels, num_vals):
    #Randomness Generator
    num_total = len(data)
    
    assert num_total == len(labels), "Incorrect Data Size"
    assert num_vals >= 0, "Partion?"
    
    #For spam data partioning since its a decimal 0.2 instead of a value like 10k
    if num_vals < 1:
        num_vals = int(num_total * num_vals)
        
    num_train_data = num_total - num_vals
    random_index = np.random.permutation(num_total)
    
    #Training Indeces
    train_indices = random_index[:num_train_data]
    val_indices = random_index[num_train_data:]
    
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

# #MNIST Plot
MNIST_loaded_data = np.load("/Users/dankim/Downloads/Cal Stuff/Cal Coursework F2020-Sp2025/Cal Berkeley Spring 2025/CS 189/Homework/hw1/data/mnist-data.npz")
m_training_data = MNIST_loaded_data["training_data"]
m_training_data = m_training_data.reshape(m_training_data.shape[0], -1)
m_training_labels = MNIST_loaded_data["training_labels"]
m_num_vals = 10000

pM_train_data, pM_train_labels, pM_validation_data, pM_validation_labels = data_partitioning(m_training_data, m_training_labels, m_num_vals)

#Train SVC Model for SVM
p_train_plt_data, p_validation_plt_data = individual_SVM_training(ms_training_size, pM_train_data, pM_train_labels, pM_validation_data, pM_validation_labels)

# MNIST_plot = plot_classification_accuracy(ms_training_size, p_train_plt_data, p_validation_plt_data, "MNIST Classification Accuracy Results")
# MNIST_plot

spam_loaded_data = np.load("/Users/dankim/Downloads/Cal Stuff/Cal Coursework F2020-Sp2025/Cal Berkeley Spring 2025/CS 189/Homework/hw1/data/spam-data.npz")
s_training_data = spam_loaded_data["training_data"]
s_training_data = s_training_data.reshape(s_training_data.shape[0], -1)
s_training_labels = spam_loaded_data["training_labels"]
s_num_vals = 0.2

sM_train_data, sM_train_labels, sM_validation_data, sM_validation_labels = data_partitioning(s_training_data, s_training_labels, s_num_vals)

# Question 5: Hyperparameter Tuning

# c_values = [1e-08, 5e-08, 1e-07, 5e-07, 1e-06, 5e-06, 1e-05, 5e-04]

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

# m_validation_accuracy, res = c_val_test(pM_train_data, pM_train_labels, pM_validation_data, pM_validation_labels, c_values)
# plot_c_values_validation(c_values, m_validation_accuracy)
# print(res)

#Question 6: K-Fold Cross-Validation

c_values_spam = [0.01, 0.1, 1, 10, 100]

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
            assert x_train.shape[0] + x_val.shape[0] == total_num
            y_train = np.concatenate(y_vals[:i] + y_vals[i+1:], axis = 0)
            assert y_train.shape[0] + y_val.shape[0] == total_num
            print(C, i, "here")
            k_fold_model = c_val_model(x_train[:10000], y_train[:10000], C)
            print(C, i, "here2")
            predict = k_fold_model.predict(x_val)
            print(C, i, "here3")
            validation_values.append(evaluation_metric(predict, y_val))
        print(C, np.mean(validation_values))
        vali_vals.append(np.mean(validation_values))
        res.append([C, np.mean(validation_values)])
        vali_vals = validation_values
    return res, vali_vals
    
k_res, k_validation_accuracy = k_fold_cross_validation(5, sM_train_data, sM_train_labels, c_values_spam)
# plot_c_values_validation(c_values_spam, k_validation_accuracy)
print(k_res)
print(k_validation_accuracy)

    
