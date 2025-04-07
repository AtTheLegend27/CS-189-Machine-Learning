import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#My Decision Tree Code Implementation
from decision_tree import DecisionTree

np.random.seed(246810)

#Code for Question 4.5


def evaluate_tree_depths_spam(max_depths):

    features = [
        "pain", "private", "bank", "money", "drug", "spam", "prescription",
        "creative", "height", "featured", "differ", "width", "other",
        "energy", "business", "message", "volumes", "revision", "path",
        "meter", "memo", "planning", "pleased", "record", "out",
        "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
        "square_bracket", "ampersand"
    ]
    
    #spam data
    path_train = 'datasets/spam_data/spam_data.mat'
    data = scipy.io.loadmat(path_train)
    X = data['training_data']
    y = np.squeeze(data['training_labels'])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=246810)
    
    #res
    train_accuracies = []
    val_accuracies = []
    
    for depth in max_depths:
        
        dt = DecisionTree(max_depth=depth, feature_labels=features)
        dt.fit(X_train, y_train)

        y_train_pred = np.array([dt._predict_sample(x) for x in X_train])
        y_val_pred = np.array([dt._predict_sample(x) for x in X_val])

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
    
    #best depth
    best_depth_idx = np.argmax(val_accuracies)
    best_depth = max_depths[best_depth_idx]
    best_val_acc = val_accuracies[best_depth_idx]
    
    print(f"\nBest max_depth = {best_depth} with validation accuracy of {best_val_acc:.4f}")
    
    # Plot results
    plt.figure()
    plt.plot(max_depths, train_accuracies, 'o-', label='Training Accuracy')
    plt.plot(max_depths, val_accuracies, 'o-', label='Validation Accuracy')
    plt.axvline(x=best_depth, color='r', linestyle='--')
    plt.xlabel('Maximum Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Performance on Spam Dataset vs. Maximum Depth')
    plt.legend()
    plt.show()

    
max_depths = range(1, 41)
results = evaluate_tree_depths_spam(max_depths)
