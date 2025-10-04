"""
To prepare the starter code, copy this file over to decision_tree_starter.py
and go through and handle all the inline TODOs.
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import pandas as pd
from pydot import graph_from_dot_data
import io

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def entropy(y):
        # TODO Comp
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy


    @staticmethod
    def information_gain(X, y, thresh):
        # TODO Comp
        if len(y) == 0:
            return 0
        
        parent_entropy = DecisionTree.entropy(y)
        
        left_indices = X < thresh
        right_indices = ~left_indices
        
        y_left = y[left_indices]
        y_right = y[right_indices]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
        
        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)
        
        left_entropy = DecisionTree.entropy(y_left)
        right_entropy = DecisionTree.entropy(y_right)
        avg_entropy = left_weight * left_entropy + right_weight * right_entropy
        
        information_gain = parent_entropy - avg_entropy
        return information_gain


    def split(self, X, y, thresh, f_idx):
        """
        Split the dataset into two subsets, given a feature and a threshold.
        Return X_0, y_0, X_1, y_1
        where (X_0, y_0) are the subset of examples whose feature_idx-th feature
        is less than thresh, and (X_1, y_1) are the other examples.
        """
        # TODO Comp
        l_idx = X[:, f_idx] < thresh
        r_idx = ~l_idx
        
        X_l, y_l = X[l_idx], y[l_idx]
        X_r, y_r = X[r_idx], y[r_idx]
        
        return X_l, X_r, y_l, y_r
        

    def fit(self, X, y):
        # TODO Comp
        
        #store lead node w training data
        self.data = X
        self.labels = y
        
        #use mode for prediction based on frequent label or no prediction
        if len(y) > 0:
            self.pred = stats.mode(y, keepdims=False)[0]
        else:
            self.pred = 0  
            
        #stop condition: max depth or pure leaf
        if self.max_depth == 0 or len(np.unique(y)) == 1:
            return
            
        best_gain = -1
        best_feature = None
        best_thresh = None
        
        n_samples, n_features = X.shape
        
        for feature_idx in range(n_features):
            
            feature_values = np.unique(X[:, feature_idx])
            
            for i in range(len(feature_values) - 1):
                thresh = (feature_values[i] + feature_values[i + 1]) / 2
                
                #information gain
                feature_col = X[:, feature_idx]
                gain = self.information_gain(feature_col, y, thresh)
                
                #update best split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_thresh = thresh
        
        #no good split = make this a leaf node
        if best_feature is None or best_thresh is None or best_gain <= 0:
            return
            
        X_left, X_right, y_left, y_right = self.split(X, y, best_thresh, best_feature)
        
        #child nodes
        if len(y_left) > 0 and len(y_right) > 0:
            self.split_idx = best_feature
            self.thresh = best_thresh

            #left tree
            self.left = DecisionTree(max_depth=self.max_depth-1, feature_labels=self.features)
            self.left.fit(X_left, y_left)
            
            #right tree
            self.right = DecisionTree(max_depth=self.max_depth-1, feature_labels=self.features)
            self.right.fit(X_right, y_right)
            
            
    def predict(self, X):
        # TODO Comp
        #leaf return pred
        if self.left is None or self.right is None:
            return self.pred
            
        #no leaf = go left or right depending on the split
        if X[self.split_idx] < self.thresh:
            return self.left._predict_sample(X)
        else:
            return self.right._predict_sample(X)
    
    #Added for batch
    def _predict_sample(self, X):
        if self.left is None or self.right is None:
            return self.pred
        
        if X[self.split_idx] < self.thresh:
            return self.left._predict_sample(X)
        else:
            return self.right._predict_sample(X)


    def _to_graphviz(self, node_id):
        if self.max_depth == 0 or self.left is None or self.right is None:
            return f'{node_id} [label="Prediction: {self.pred}\nSamples: {self.labels.size}"];\n'
        else:
            graph = f'{node_id} [label="{self.features[self.split_idx]} < {self.thresh:.2f}"];\n'
            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2
            if self.left is not None:
                graph += f'{node_id} -> {left_id};\n'
                graph += self.left._to_graphviz(left_id)
            if self.right is not None:
                graph += f'{node_id} -> {right_id};\n'
                graph += self.right._to_graphviz(right_id)
            return graph

    def to_graphviz(self):
        graph = "digraph Tree {\nnode [shape=box];\n"
        graph += self._to_graphviz(0)
        graph += "}\n"
        return graph
        
    def __repr__(self):
        if self.max_depth == 0 or self.left is None or self.right is None:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO Comp
        n_samples = X.shape[0]
        
        for tree in self.decision_trees:
            #bootstrap sample (random sampling with replacement)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            #fit bootstrap sample tree
            tree.fit(X_bootstrap, y_bootstrap)
            
        return self
    
    def predict(self, X):
        # TODO Comp
        
        tree_predictions = np.array([tree.predict(X) for tree in self.decision_trees])
        
        majority_votes = stats.mode(tree_predictions, axis=0, keepdims=False)[0]
        
        return majority_votes



class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)



def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        # TODO Comp
        for col in range(data.shape[1]):
            #replace w mode
            missing_indices = np.where(data[:, col] == -1)[0]
            if len(missing_indices) == 0:
                continue
                
            valid_indices = np.where(data[:, col] != -1)[0]
            if len(valid_indices) == 0:
                continue
                
            mode_val = stats.mode(data[valid_indices, col], keepdims=False)[0]
            
            data[missing_indices, col] = mode_val
            
    return data, onehot_features



def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def generate_submission(testing_data, predictions, dataset="titanic"):
    assert dataset in ["titanic", "spam"], f"dataset should be either 'titanic' or 'spam'"
    # This code below will generate the predictions.csv file.
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(f'predictions_{dataset}.csv', index_label='Id')


if __name__ == "__main__":
    dataset = "titanic"
    # dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    # Decision Tree
    print("\n\nDecision Tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)

    # Visualize Decision Tree
    print("\n\nTree Structure")
    # Print using repr
    print(dt.__repr__())
    # Save tree to pdf
    graph_from_dot_data(dt.to_graphviz())[0].write_pdf("%s-basic-tree.pdf" % dataset)

    # Random Forest
    print("\n\nRandom Forest")
    rf = RandomForest(params, n=N, m=np.int_(np.sqrt(X.shape[1])))
    rf.fit(X, y)
    evaluate(rf)

    # Generate Test Predictions
    print("\n\nGenerate Test Predictions")
    pred = rf.predict(Z)
    generate_submission(Z, pred, dataset)
 
 
 
 
