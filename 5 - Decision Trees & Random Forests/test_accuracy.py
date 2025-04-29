import numpy as np
from numpy import genfromtxt
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#My Decision Tree Code Implementation
from decision_tree import DecisionTree, BaggedTrees, RandomForest, preprocess

np.random.seed(246810)


#Code for Question 4.4
def evaluate_model_performance(model, X_train, y_train, X_val, y_val):
    
    model.fit(X_train, y_train)
    
    #predictions
    if hasattr(model, "_predict_sample"):
        y_train_pred = np.array([model._predict_sample(x) for x in X_train])
        y_val_pred = np.array([model._predict_sample(x) for x in X_val])
    else:
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
    
    #calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    return {
        "training_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
    }


def evaluate_titanic_dataset():
    
    path_train = 'datasets/titanic/titanic_training.csv'
    data = genfromtxt(path_train, delimiter=',', dtype=None)
    
    #labels
    y = data[1:, 0]  
    labeled_idx = np.where(y != b'')[0]
    y = np.array(y[labeled_idx], dtype=float).astype(int)
    
    #preprocess
    X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
    X = X[labeled_idx, :]
    features = list(data[0, 1:]) + onehot_features
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=246810)

    #model
    dt = DecisionTree(max_depth=5, feature_labels=features)
    
    params = {
        "max_depth": 5,
        "min_samples_leaf": 10,
    }
    rf = RandomForest(params, n=100, m=int(np.sqrt(X.shape[1])))
    
    #evaluate models
    dt_results = evaluate_model_performance(dt, X_train, y_train, X_val, y_val)
    rf_results = evaluate_model_performance(rf, X_train, y_train, X_val, y_val)
    
    return {
        "decision_tree": dt_results,
        "random_forest": rf_results
    }

def evaluate_spam_dataset():
    features = [
        "pain", "private", "bank", "money", "drug", "spam", "prescription",
        "creative", "height", "featured", "differ", "width", "other",
        "energy", "business", "message", "volumes", "revision", "path",
        "meter", "memo", "planning", "pleased", "record", "out",
        "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
        "square_bracket", "ampersand"
    ]
    
    path_train = 'datasets/spam_data/spam_data.mat'
    data = scipy.io.loadmat(path_train)
    X = data['training_data']
    y = np.squeeze(data['training_labels'])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=246810)
    
    #decision tree
    dt = DecisionTree(max_depth=5, feature_labels=features)
    
    params = {
        "max_depth": 5,
        "min_samples_leaf": 10,
    }
    
    #random forest
    rf = RandomForest(params, n=100, m=int(np.sqrt(X.shape[1])))
    
    #evaluate models
    dt_results = evaluate_model_performance(dt, X_train, y_train, X_val, y_val)
    rf_results = evaluate_model_performance(rf, X_train, y_train, X_val, y_val)
    
    return {
        "decision_tree": dt_results,
        "random_forest": rf_results
    }

print("titanic")
titanic_results = evaluate_titanic_dataset()
print("spam")
spam_results = evaluate_spam_dataset()

print(titanic_results, spam_results)