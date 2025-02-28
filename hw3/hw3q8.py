#Question 8: Gaussian Classification for Digits and Spam

#Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster
import pandas as pd


#My implementation of LDA/QDA
from lda_code import LDA
from qda_code import QDA


np.random.seed(7)


#Loading MNIST Data
mnist_data =  np.load("/Users/dankim/Downloads/Cal Stuff/Cal Coursework F2020-Sp2025/Cal Berkeley Spring 2025/CS 189/Homework/hw3/data/mnist-data-hw3.npz")


m_training_data = mnist_data["training_data"]
m_training_data_norm = scipy.cluster.vq.whiten(m_training_data)

m_training_labels = mnist_data["training_labels"]
m_unique_labels = np.unique(m_training_labels)


#Question 8 Part 1
def mnist_training(t_labels_unique, t_labels, t_data_norm):
    curr_trained = {}
    
    for label in t_labels_unique:
       idx = (t_labels == label).flatten()
       curr_data = t_data_norm[idx]
       curr_data = curr_data.reshape(curr_data.shape[0], -1)
       u = np.mean(curr_data, axis = 0)
       ep = np.cov(curr_data, rowvar = False)
       curr_trained[label] = (u, ep)
    
    return curr_trained

mnist_trained = mnist_training(m_unique_labels, m_training_labels, m_training_data_norm)


#Question 8 Part 2

def mnist_plot_cov(t_labels_unique, t_labels, t_data_norm):
    idx = (t_labels == t_labels_unique[0]).flatten()
    curr_data = t_data_norm[idx]
    curr_data = curr_data.reshape(curr_data.shape[0], -1)
    cov_m = np.corrcoef(curr_data, rowvar = False)
    cov_m[np.isnan(cov_m)] = 0
    cov_m_abs = np.abs(cov_m)
    cax = plt.imshow(cov_m_abs)
    plt.colorbar(cax, label='Covariance Strength')
    plt.title("Question 8 Part 2: Covariance Matrix Visualization for Digit 0")
    plt.xlabel("Pixel Index")
    plt.ylabel("Pixel Index")
    plt.show()
    
mnist_plot_cov(m_unique_labels,m_training_labels, m_training_data_norm)


#Question 8 Part 3

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


m_train_data, m_train_labels, m_validation_data, m_validation_labels =  data_partitioning(m_training_data_norm, m_training_labels, num_vals = 10000)

rand_training_points = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
m_validation_labels.flatten()
lda_errors = []
qda_errors = []


lda_digit_errors = {digit: [] for digit in m_unique_labels}
qda_digit_errors = {digit: [] for digit in m_unique_labels}



#Function to evaluate the classifiers
def evaluate_classifier(classifier, tst_data, tst_labels):
    if classifier == lda:
        predictions = classifier.lda_classify(tst_data)
    else:
        predictions = classifier.qda_classify(tst_data)
    accuracy = np.mean(predictions == tst_labels)
    error_rate = 1 - accuracy
    return error_rate


#Training
for size in rand_training_points:
    subset_data = m_train_data[:size]
    subset_labels = m_train_labels[:size]
    
    lda = LDA(m_unique_labels)
    lda.lda_training(subset_labels, subset_data)
    
    qda = QDA(m_unique_labels)
    qda.qda_training(subset_labels, subset_data)
    
    lda_errors.append(evaluate_classifier(lda, m_validation_data, m_validation_labels))    
    qda_errors.append(evaluate_classifier(qda, m_validation_data, m_validation_labels))
    
    for digit in m_unique_labels:
        idxes = (m_validation_labels == digit).flatten()
        digit_data = m_validation_data[idxes]
        digit_labels = m_validation_labels[idxes]
        
        lda_digit_errors[digit].append(evaluate_classifier(lda, digit_data, digit_labels))
        qda_digit_errors[digit].append(evaluate_classifier(qda, digit_data, digit_labels))
    
    
    
#Part A/B: Plotting LDA and GDA Comparison

plt.plot(rand_training_points, lda_errors, label="LDA Error")
plt.plot(rand_training_points, qda_errors, label="QDA Error")
plt.xlabel("Training Set Size")
plt.ylabel("Error Rate")
plt.title("LDA vs. QDA Error Rates")
plt.legend()
plt.show()


#Part D: Plotting LDA/GDA classification per digit

# Plot LDA classification per digit
for digit in m_unique_labels:
    plt.plot(rand_training_points, lda_digit_errors[digit], label=f"Digit {digit}")
plt.xlabel("Training Set Size")
plt.ylabel("Error Rate")
plt.title("LDA: Error Rate for Each Digit")
plt.legend()
plt.show()

# Plot QDA classification per digit
for digit in m_unique_labels:
    plt.plot(rand_training_points, qda_digit_errors[digit], label=f"Digit {digit}")
plt.xlabel("Training Set Size")
plt.ylabel("Error Rate")
plt.title("QDA: Error Rate for Each Digit")
plt.legend()
plt.show()


# Question 4/5: Kaggle:

#Kaggle CSV function from HW 1
def results_to_csv(y_test, file_name): 
    y_test = y_test.astype(int) 
    df = pd.DataFrame({"Category": y_test}) 
    df.index += 1 
    df.to_csv(file_name, index_label="Id")
    
    

#Question 4: MNIST: 
m_test_data = mnist_data["test_data"]
m_test_data_norm = scipy.cluster.vq.whiten(m_test_data)

# Train on the full training set
lda = LDA(m_unique_labels)
lda.lda_training(m_training_labels, m_training_data_norm)

# qda = QDA(m_unique_labels)
# qda.qda_training(m_training_labels, m_training_data_norm)

lda_predictions = lda.lda_classify(m_test_data_norm)
# qda_predictions = qda.qda_classify(m_test_data_norm)

better_prediction = lda_predictions

results_to_csv(better_prediction, "MNIST_Test_Labels.csv") 



#Question 5: SPAM:
spam_data = np.load("/Users/dankim/Downloads/Cal Stuff/Cal Coursework F2020-Sp2025/Cal Berkeley Spring 2025/CS 189/Homework/hw3/data/spam-data-hw3.npz")

s_training_data = spam_data["training_data"]
s_training_data_norm = scipy.cluster.vq.whiten(s_training_data)

s_training_labels = spam_data["training_labels"]
s_unique_labels = np.unique(s_training_labels)

s_test_data = spam_data["test_data"]
s_test_data_norm = scipy.cluster.vq.whiten(s_test_data)

# Train on the full training set
s_lda = LDA(s_unique_labels)
s_lda.lda_training(s_training_labels, s_training_data_norm)

# s_qda = QDA(s_unique_labels)
# s_qda.qda_training(s_training_labels, s_training_data_norm)

s_lda_predictions = s_lda.lda_classify(s_test_data_norm)
# s_qda_predictions = s_qda.qda_classify(s_test_data_norm)

s_better_prediction = s_lda_predictions

# Save Spam Predictions to CSV
results_to_csv(s_better_prediction, "spam_Test_Labels.csv")