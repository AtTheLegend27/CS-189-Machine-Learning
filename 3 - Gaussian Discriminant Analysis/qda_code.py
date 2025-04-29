#QDA Class

#Imports
import numpy as np
from scipy.stats import multivariate_normal


class QDA: 
    def __init__(self, t_labels_unique):
        self.t_labels_unique = t_labels_unique
        self.means = {}
        self.total_ep = {}
        self.priors = {}  

    
    def qda_training(self, t_labels, t_data):
        total_samples = len(t_labels)
        
        if len(t_data.shape) > 2:
            t_data = t_data.reshape(t_data.shape[0], -1)
            
        for label in self.t_labels_unique:
            idx = (t_labels == label).flatten()
            class_data = t_data[idx]
            
            if len(class_data.shape) > 2:
                class_data = class_data.reshape(class_data.shape[0], -1)
                
            self.means[label] = np.mean(class_data, axis=0)
            curr_ep = np.cov(class_data, rowvar=False)
            
            epsilon = 1e-5
            curr_ep += epsilon * np.eye(curr_ep.shape[0])
            
            self.total_ep[label] = curr_ep      
            self.priors[label] = idx.sum() / total_samples
    
    
    def qda_classify(self, tst_data):
        
        if len(tst_data.shape) > 2:
            tst_data = tst_data.reshape(tst_data.shape[0], -1)
        
        predictions = np.zeros(tst_data.shape[0], dtype=self.t_labels_unique.dtype)
        
        # Multivariate normal distributions for each class
        distributions = {
            label: multivariate_normal(mean=self.means[label], cov=self.total_ep[label])
            for label in self.t_labels_unique
        }
        
        log_probs = np.zeros((tst_data.shape[0], len(self.t_labels_unique)))
        
        for i, label in enumerate(self.t_labels_unique):
            log_probs[:, i] = distributions[label].logpdf(tst_data)
        
        max_indices = np.argmax(log_probs, axis=1)
        predictions = self.t_labels_unique[max_indices]
        
        return predictions
        